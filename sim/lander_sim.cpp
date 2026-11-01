// lander_sim: whole flights of the frozen task (lander/env.py + gymnasium LunarLander 0.29.1) in C++, for training only.
//
// Box2D is the 2.3.x source that box2d-py 2.3.5 wraps (sim/Box2D, unchanged). The world is rebuilt per flight in the
// order gymnasium's reset() builds it (moon, lander, legs, joints, the initial push, the reset's step(0), then our
// boulder). Every random number (terrain, initial push, engine dispersion, gust phase) is drawn in Python with
// gymnasium's own RNG and passed in, so nothing random is ported. The arithmetic mirrors the Python step for step:
// doubles where Python computes in float64, float32 where Box2D or numpy's float32 observations do.
// The NEAT network is evaluated in C++ too (NeatJsonNet's flat arrays), so one call flies a batch with no Python.
//
// It is a fast copy, not the task: lander/cppenv.py documents the equivalence test, and checks, selection and the
// goal test always fly the reference Python env (BRIEF section 6).

#include <Box2D/Box2D.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cmath>
#include <cstdint>
#include <vector>

namespace py = pybind11;

static const double SCALE = 30.0, FPS = 50.0;
static const double VW = 600.0, VH = 400.0;
static const double W = VW / SCALE, H = VH / SCALE;
static const double MAIN_ENGINE_POWER = 13.0, SIDE_ENGINE_POWER = 0.6;
static const int MAX_STEPS = 1000;
static const double WIND_POWER = 15.0;
static const double DRIFT_SCALE = 1 / (0.0008 * WIND_POWER), DRIFT_RATE = 0.15;
static const double MAIN_PUSH = -0.07135, WIND_RATE = 0.3, WIND_MAIN_WEIGHT = 0.1;
static const double PAD_HALF_WIDTH = 0.2;

struct Listener : b2ContactListener {
    b2Body *lander = nullptr, *legs[2] = {nullptr, nullptr};
    bool game_over = false, contact[2] = {false, false};
    void BeginContact(b2Contact *c) override {
        b2Body *a = c->GetFixtureA()->GetBody(), *b = c->GetFixtureB()->GetBody();
        if (lander == a || lander == b) game_over = true;
        for (int i = 0; i < 2; i++)
            if (legs[i] == a || legs[i] == b) contact[i] = true;
    }
    void EndContact(b2Contact *c) override {
        b2Body *a = c->GetFixtureA()->GetBody(), *b = c->GetFixtureB()->GetBody();
        for (int i = 0; i < 2; i++)
            if (legs[i] == a || legs[i] == b) contact[i] = false;
    }
};

struct Net {            // NeatJsonNet.flat_i / flat_f, read the same way as neatnet._act
    const int64_t *fi; double *ff;
    int act(const double *x) const {
        int64_t n_in = fi[0], m = fi[1], n_out = fi[2];
        int64_t rows = 3, tanh_f = 3 + m, start = 3 + 2 * m, src = start + m + 1, out_idx = src + fi[start + m];
        int64_t bias = n_in, wt = n_in + m, v = wt + fi[start + m];
        for (int64_t i = 0; i < n_in; i++) ff[v + i] = x[i] / ff[i];
        for (int64_t j = 0; j < m; j++) {
            double s = ff[bias + j];
            for (int64_t c = fi[start + j]; c < fi[start + j + 1]; c++) s += ff[wt + c] * ff[v + fi[src + c]];
            ff[v + fi[rows + j]] = fi[tanh_f + j] ? std::tanh(s) : s;
        }
        int best = 0;
        for (int o = 1; o < n_out; o++)
            if (ff[v + fi[out_idx + o]] > ff[v + fi[out_idx + best]]) best = o;
        return best;
    }
};

// one flight's set-up, from cppenv.flight_spec (Python)
struct Spec {
    const double *smooth_y;   // 11 terrain heights
    double fx, fy;            // the reset's initial random force
    long wind_idx, torque_idx;
    double wind_power, turbulence_power; bool enable_wind;
    double push, support;
    bool boulder; double bx, by, br;
    const double *disp;       // 2 + 2 * MAX_STEPS uniforms(-1, 1): the reset's step(0), then one pair per step
    double side_margin, top_margin;
};

struct Result { double reward; int steps; int outcome; double x_final, touch_speed, clearance; };
enum { CRASHED = 0, LANDED = 1, TIMEOUT = 2, BOULDER = 3 };

struct Flight {
    b2World world{b2Vec2(0.0f, -10.0f)};
    Listener listener;
    b2Body *moon, *lander, *legs[2], *boulder = nullptr;
    const Spec &s;
    double helipad_y = H / 4;
    double prev_shaping = 0; bool has_prev = false;
    long wind_idx, torque_idx;
    float obs[8];

    explicit Flight(const Spec &spec) : s(spec), wind_idx(spec.wind_idx), torque_idx(spec.torque_idx) {
        world.SetContactListener(&listener);
        const int CHUNKS = 11;
        double chunk_x[CHUNKS];
        for (int i = 0; i < CHUNKS; i++) chunk_x[i] = W / (CHUNKS - 1) * i;
        // moon: CreateStaticBody(shapes=edgeShape([(0, 0), (W, 0)])) with pybox2d's default b2FixtureDef
        b2BodyDef bd;
        moon = world.CreateBody(&bd);
        {
            b2EdgeShape e; e.Set(b2Vec2(0.0f, 0.0f), b2Vec2((float)W, 0.0f));
            b2FixtureDef fd; fd.shape = &e; moon->CreateFixture(&fd);
        }
        for (int i = 0; i < CHUNKS - 1; i++) {
            b2EdgeShape e;
            e.Set(b2Vec2((float)chunk_x[i], (float)s.smooth_y[i]), b2Vec2((float)chunk_x[i + 1], (float)s.smooth_y[i + 1]));
            b2FixtureDef fd; fd.shape = &e; fd.density = 0; fd.friction = 0.1f; moon->CreateFixture(&fd);
        }
        double initial_y = VH / SCALE, initial_x = VW / SCALE / 2;
        {
            b2BodyDef ld; ld.type = b2_dynamicBody; ld.position.Set((float)initial_x, (float)initial_y); ld.angle = 0.0f;
            lander = world.CreateBody(&ld);
            const int P[6][2] = {{-14, 17}, {-17, 0}, {-17, -10}, {17, -10}, {17, 0}, {14, 17}};
            b2Vec2 v[6];
            for (int i = 0; i < 6; i++) v[i].Set((float)(P[i][0] / SCALE), (float)(P[i][1] / SCALE));
            b2PolygonShape poly; poly.Set(v, 6);
            b2FixtureDef fd; fd.shape = &poly; fd.density = 5.0f; fd.friction = 0.1f;
            fd.filter.categoryBits = 0x0010; fd.filter.maskBits = 0x001; fd.restitution = 0.0f;
            lander->CreateFixture(&fd);
        }
        lander->ApplyForceToCenter(b2Vec2((float)s.fx, (float)s.fy), true);
        for (int k = 0; k < 2; k++) {
            int i = k == 0 ? -1 : 1;
            b2BodyDef gd; gd.type = b2_dynamicBody;
            gd.position.Set((float)(initial_x - i * 20 / SCALE), (float)initial_y); gd.angle = (float)(i * 0.05);
            b2Body *leg = world.CreateBody(&gd);
            b2PolygonShape box; box.SetAsBox((float)(2 / SCALE), (float)(8 / SCALE));
            b2FixtureDef fd; fd.shape = &box; fd.density = 1.0f; fd.restitution = 0.0f;
            fd.filter.categoryBits = 0x0020; fd.filter.maskBits = 0x001;
            leg->CreateFixture(&fd);
            b2RevoluteJointDef j;
            j.bodyA = lander; j.bodyB = leg;
            j.localAnchorA.Set(0.0f, 0.0f); j.localAnchorB.Set((float)(i * 20 / SCALE), (float)(18 / SCALE));
            j.enableMotor = true; j.enableLimit = true; j.maxMotorTorque = 40.0f; j.motorSpeed = (float)(0.3 * i);
            if (i == -1) { j.lowerAngle = (float)(+0.9 - 0.5); j.upperAngle = 0.9f; }
            else { j.lowerAngle = -0.9f; j.upperAngle = (float)(-0.9 + 0.5); }
            j.referenceAngle = (float)((double)leg->GetAngle() - (double)lander->GetAngle());   // pybox2d sets it
            world.CreateJoint(&j);
            legs[k] = leg;
        }
        listener.lander = lander; listener.legs[0] = legs[0]; listener.legs[1] = legs[1];
        double r;
        step(0, s.disp[0], s.disp[1], r);              // the reset ends with step(0)
        if (s.boulder) {
            b2BodyDef bb; bb.position.Set((float)s.bx, (float)s.by);
            boulder = world.CreateBody(&bb);
            b2CircleShape c; c.m_radius = (float)s.br;
            b2FixtureDef fd; fd.shape = &c; fd.density = 0; fd.friction = 0.3f;
            boulder->CreateFixture(&fd);
        }
    }

    bool touching() const { return listener.contact[0] || listener.contact[1]; }

    // fastenv.fast_step / gymnasium LunarLander.step for a discrete action
    bool step(int action, double d0, double d1, double &reward) {
        if (s.enable_wind && !touching()) {
            double wind_mag = std::tanh(std::sin(0.02 * wind_idx) + (std::sin(M_PI * 0.01 * wind_idx))) * s.wind_power;
            wind_idx += 1;
            lander->ApplyForceToCenter(b2Vec2((float)wind_mag, 0.0f), true);
            double torque_mag = std::tanh(std::sin(0.02 * torque_idx) + (std::sin(M_PI * 0.01 * torque_idx))) * s.turbulence_power;
            torque_idx += 1;
            lander->ApplyTorque((float)torque_mag, true);
        }
        double angle = lander->GetAngle();
        double tip0 = std::sin(angle), tip1 = std::cos(angle);
        double side0 = -tip1, side1 = tip0;
        double disp0 = d0 / SCALE, disp1 = d1 / SCALE;
        double m_power = 0.0, s_power = 0.0;
        if (action == 2) {
            m_power = 1.0;
            double ox = tip0 * (4 / SCALE + 2 * disp0) + side0 * disp1;
            double oy = -tip1 * (4 / SCALE + 2 * disp0) - side1 * disp1;
            b2Vec2 p = lander->GetPosition();
            b2Vec2 ip((float)((double)p.x + ox), (float)((double)p.y + oy));
            lander->ApplyLinearImpulse(b2Vec2((float)(-ox * MAIN_ENGINE_POWER * m_power), (float)(-oy * MAIN_ENGINE_POWER * m_power)), ip, true);
        }
        if (action == 1 || action == 3) {
            int direction = action - 2;
            s_power = 1.0;
            double away = (direction * 12) / SCALE;
            double ox = tip0 * disp0 + side0 * (3 * disp1 + away);
            double oy = -tip1 * disp0 - side1 * (3 * disp1 + away);
            b2Vec2 p = lander->GetPosition();
            b2Vec2 ip((float)((double)p.x + ox - tip0 * 17 / SCALE), (float)((double)p.y + oy + tip1 * 14 / SCALE));
            lander->ApplyLinearImpulse(b2Vec2((float)(-ox * SIDE_ENGINE_POWER * s_power), (float)(-oy * SIDE_ENGINE_POWER * s_power)), ip, true);
        }
        world.Step((float)(1.0 / FPS), 6 * 30, 2 * 30);
        b2Vec2 pos = lander->GetPosition(), vel = lander->GetLinearVelocity();
        double st[8] = {
            ((double)pos.x - VW / SCALE / 2) / (VW / SCALE / 2),
            ((double)pos.y - (helipad_y + 18 / SCALE)) / (VH / SCALE / 2),
            (double)vel.x * (VW / SCALE / 2) / FPS,
            (double)vel.y * (VH / SCALE / 2) / FPS,
            (double)lander->GetAngle(),
            20.0 * (double)lander->GetAngularVelocity() / FPS,
            listener.contact[0] ? 1.0 : 0.0,
            listener.contact[1] ? 1.0 : 0.0,
        };
        double shaping = -100 * std::sqrt(st[0] * st[0] + st[1] * st[1]) - 100 * std::sqrt(st[2] * st[2] + st[3] * st[3])
                         - 100 * std::fabs(st[4]) + 10 * st[6] + 10 * st[7];
        reward = 0;
        if (has_prev) reward = shaping - prev_shaping;
        prev_shaping = shaping; has_prev = true;
        reward -= m_power * 0.30;
        reward -= s_power * 0.03;
        bool terminated = false;
        if (listener.game_over || std::fabs(st[0]) >= 1.0) { terminated = true; reward = -100; }
        if (!lander->IsAwake()) { terminated = true; reward = +100; }
        for (int i = 0; i < 8; i++) obs[i] = (float)st[i];
        return terminated;
    }

    bool boulder_hit() const {
        b2Body *bodies[3] = {lander, legs[0], legs[1]};
        for (b2Body *b : bodies)
            for (b2ContactEdge *ce = b->GetContactList(); ce; ce = ce->next)
                if (ce->other == boulder && ce->contact->IsTouching()) return true;
        return false;
    }

    // env.waypoint_inputs -> teacher.target_x, with the route parameters
    void waypoint(const double *v, double &a, double &b) const {
        double x = v[0], y = v[1], dx = v[10], dy = v[11], r = v[12];
        a = x - 0.0; b = 0.0;
        if (r <= 0) return;
        double bx = x + dx, by = y + dy, ry = r * 1.5;
        if (y < by - ry - 0.15) return;
        double side = (x - bx) > 0 ? 1.0 : -1.0;
        if (std::fabs(bx) < r + 0.25 && std::fabs(x - bx) < 0.05) side = bx <= 0 ? 1.0 : -1.0;
        double tx = bx + side * (r + s.side_margin * 0.12 + 0.06);
        a = x - tx;
        if (std::fabs(x - bx) < r + 0.1) b = y - (by + ry + s.top_margin * 0.12);
    }

    Result fly(const Net &net) {
        Result out{0.0, 0, -1, 0.0, NAN, INFINITY};
        double drift = 0.0, wind = 0.0, prev_vx = 0.0; bool has_prev_vx = false;
        int action = 0;
        double x[16];
        for (int t = 0; ; ) {
            // Flight.inputs()
            for (int i = 0; i < 8; i++) x[i] = obs[i];
            x[8] = drift; x[9] = std::fabs((double)obs[0]);
            if (boulder) {
                b2Vec2 p = lander->GetPosition();
                x[10] = (s.bx - (double)p.x) / (W / 2); x[11] = (s.by - (double)p.y) / (H / 2); x[12] = s.br / (W / 2);
            } else x[10] = x[11] = x[12] = 0.0;
            waypoint(x, x[13], x[14]);
            x[15] = wind;
            action = net.act(x);
            // Flight.step()
            bool was_touching = touching();
            if (s.push != 0.0 && !was_touching) lander->ApplyForceToCenter(b2Vec2((float)s.push, 0.0f), true);
            if (s.support != 0.0) {
                double tq = -s.support * (8.0 * (double)lander->GetAngle() + 2.0 * (double)lander->GetAngularVelocity());
                lander->ApplyTorque((float)tq, true);
            }
            double vy_before = obs[3];
            float prev[8]; for (int i = 0; i < 8; i++) prev[i] = obs[i];
            double r;
            bool terminated = step(action, s.disp[2 + 2 * t], s.disp[3 + 2 * t], r);
            // DriftSensor, WindSensor
            if (has_prev_vx && (action == 0 || action == 2) && !(obs[6] != 0.0f || obs[7] != 0.0f))
                drift += DRIFT_RATE * ((double)(float)(obs[2] - prev_vx) * DRIFT_SCALE - drift);
            prev_vx = obs[2]; has_prev_vx = true;
            if (!(prev[6] != 0.0f || prev[7] != 0.0f || obs[6] != 0.0f || obs[7] != 0.0f)) {
                double push = action == 2 ? MAIN_PUSH * std::sin((double)prev[4]) : (action == 1 ? -0.00983 : action == 3 ? 0.0098 : 0.0);
                double m = ((double)(float)(obs[2] - prev[2]) - push) / (0.0008 * WIND_POWER);
                double w = action == 2 ? WIND_MAIN_WEIGHT : 1.0;
                wind += WIND_RATE * w * (m - wind);
            }
            out.reward += r;
            t += 1; out.steps = t;
            if (!was_touching && touching() && std::isnan(out.touch_speed)) out.touch_speed = std::fabs(vy_before);
            if (boulder) {
                bool hit = boulder_hit();
                b2Vec2 p = lander->GetPosition();
                double hx = (double)p.x - s.bx, hy = (double)p.y - s.by;     // Python's math.hypot is correctly rounded
                double c = (double)std::sqrt((long double)hx * hx + (long double)hy * hy) - s.br - 0.6;
                if (c < out.clearance) out.clearance = c;
                if (hit) {
                    if (!terminated) out.reward += -100.0;
                    out.outcome = BOULDER; break;
                }
            }
            if (terminated) { out.outcome = r > 0 ? LANDED : CRASHED; break; }
            if (t >= MAX_STEPS) { out.outcome = TIMEOUT; break; }
        }
        out.x_final = obs[0];
        return out;
    }
};

// fly(flat_i, flat_f, smooth_y[n,11], scal[n,13], disp[n,2+2*MAX_STEPS]) -> res[n,6]
// scal columns: fx fy wind_idx torque_idx wind_power turbulence_power enable_wind push support boulder bx by br
static py::array_t<double> fly(py::array_t<int64_t, py::array::c_style | py::array::forcecast> fi,
                               py::array_t<double, py::array::c_style | py::array::forcecast> ff_in,
                               py::array_t<double, py::array::c_style | py::array::forcecast> smooth_y,
                               py::array_t<double, py::array::c_style | py::array::forcecast> scal,
                               py::array_t<double, py::array::c_style | py::array::forcecast> disp,
                               double side_margin, double top_margin) {
    ssize_t n = scal.shape(0);
    if (smooth_y.shape(1) != 11 || scal.shape(1) != 13 || disp.shape(1) != 2 + 2 * MAX_STEPS ||
        smooth_y.shape(0) != n || disp.shape(0) != n)
        throw std::invalid_argument("bad flight spec shapes");
    std::vector<double> ff(ff_in.data(), ff_in.data() + ff_in.size());   // node values are scratch: own copy
    Net net{fi.data(), ff.data()};
    py::array_t<double> res({n, (ssize_t)6});
    double *R = res.mutable_data();
    const double *Y = smooth_y.data(), *S = scal.data(), *D = disp.data();
    {
        py::gil_scoped_release nogil;
        for (ssize_t k = 0; k < n; k++) {
            const double *q = S + 13 * k;
            Spec sp{Y + 11 * k, q[0], q[1], (long)q[2], (long)q[3], q[4], q[5], q[6] != 0.0, q[7], q[8],
                    q[9] != 0.0, q[10], q[11], q[12], D + (2 + 2 * MAX_STEPS) * k, side_margin, top_margin};
            Flight f(sp);
            Result r = f.fly(net);
            double *o = R + 6 * k;
            o[0] = r.reward; o[1] = r.steps; o[2] = r.outcome; o[3] = r.x_final; o[4] = r.touch_speed; o[5] = r.clearance;
        }
    }
    return res;
}

PYBIND11_MODULE(lander_sim, m) {
    m.doc() = "Whole LunarLander flights (the frozen task) with the NEAT network, in C++. Training only.";
    m.def("fly", &fly);
    m.attr("MAX_STEPS") = MAX_STEPS;
}
