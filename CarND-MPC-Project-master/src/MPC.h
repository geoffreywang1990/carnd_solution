#ifndef MPC_H
#define MPC_H

#include <vector>
#include "Eigen-3.3/Eigen/Core"

using namespace std;

extern const double latency;
extern const int latency_dt_units;

// For converting back and forth between radians and degrees.
inline constexpr double pi() { return M_PI; }
inline double deg2rad(double x) { return x * pi() / 180; }
inline double rad2deg(double x) { return x * 180 / pi(); }

struct Solution {
    vector<double> X;
    vector<double> Y;
    vector<double> Delta;
    vector<double> A;
};

class MPC {
public:
    MPC();

    virtual ~MPC();

    // Solve the model given an initial state and polynomial coefficients.
    // Return the first actuatotions.
    Solution Solve(const Eigen::VectorXd &state, const Eigen::VectorXd &coeffs);

    double delta_prev = 0;
    double a_prev = 0;
};

#endif /* MPC_H */
