#include "MPC.h"
#include <cppad/cppad.hpp>
#include <cppad/ipopt/solve.hpp>
#include "Eigen-3.3/Eigen/Core"

using CppAD::AD;

// Set the timestep length and duration
const int N = 12;
const double dt = 0.05; // in seconds, 20 ms
const double latency = 0.1; // in seconds, 100 ms
const int latency_dt_units = latency / dt;

// This value assumes the model presented in the classroom is used.
//
// It was obtained by measuring the radius formed by running the vehicle in the
// simulator around in a circle with a constant steering angle and velocity on a
// flat terrain.
//
// Lf was tuned until the the radius formed by the simulating the model
// presented in the classroom matched the previous radius.
//
// This is the length from front to CoG that has a similar radius.
const double Lf = 2.67;

// Reference variables to be achieved by the optimizer
const double v_ref = 90;
const double cte_ref = 0;
const double epsi_ref = 0;

// The solver receives all state and actuator variables in one vector
// First N for x, next N for y, ..., N for epsi, N-1 for steer, N-1 for throttle
const size_t x_start = 0; // first N variables are for x
const size_t y_start = x_start + N; // next N variables are for y ...
const size_t psi_start = y_start + N;
const size_t v_start = psi_start + N;
const size_t cte_start = v_start + N;
const size_t epsi_start = cte_start + N;
const size_t delta_start = epsi_start + N;  // steer and throttle have only N-1 variables to optimize
const size_t a_start = delta_start + N - 1;

class FG_eval {
public:
    // Fitted polynomial coefficients
    Eigen::VectorXd coeffs;

    FG_eval(const Eigen::VectorXd &coeffs) {
        this->coeffs = coeffs;
    }

    typedef CPPAD_TESTVECTOR(AD<double>) ADvector;

    void operator()(ADvector& fg, const ADvector& vars) {
        // The cost is stored is the first element of `fg`.
        // Any additions to the cost should be added to `fg[0]`.
        fg[0] = 0;

        // The part of the cost based on the reference state.
        for (int t = 0; t < N; ++t) {
            // trajectory
            fg[0] += CppAD::pow(vars[cte_start + t] - cte_ref, 2);
            fg[0] += CppAD::pow(vars[epsi_start + t] - epsi_ref, 2);
            fg[0] += CppAD::pow(vars[v_start + t] - v_ref, 2) / 3;
        }

        // Minimize change-rate of actuators.
        for (int t = 0; t < N - 1; ++t) {
            fg[0] += CppAD::pow(vars[delta_start + t], 2);
            fg[0] += CppAD::pow(vars[a_start + t], 2) * 10;
        }

        // Minimize the value gap between sequential actuations.
        for (int t = 0; t < N - 2; ++t) {
            fg[0] += CppAD::pow(vars[delta_start + t + 1] - vars[delta_start + t], 2) * 500;
            fg[0] += CppAD::pow(vars[a_start + t + 1] - vars[a_start + t], 2);
        }

        // Initialize the model to the initial state.
        // fg[0] is reserved for the cost value, so the other indices are bumped up by 1.
        fg[1 + x_start] = vars[x_start];
        fg[1 + y_start] = vars[y_start];
        fg[1 + psi_start] = vars[psi_start];
        fg[1 + v_start] = vars[v_start];
        fg[1 + cte_start] = vars[cte_start];
        fg[1 + epsi_start] = vars[epsi_start];

        // the other constraints based on the vehicle model
        // The rest of the constraints
        for (int t = 1; t < N; ++t) {
            // The state at time t+1 .
            const AD<double> x1 = vars[x_start + t];
            const AD<double> y1 = vars[y_start + t];
            const AD<double> psi1 = vars[psi_start + t];
            const AD<double> v1 = vars[v_start + t];
            const AD<double> cte1 = vars[cte_start + t];
            const AD<double> epsi1 = vars[epsi_start + t];

            // The state at time t.
            const AD<double> x0 = vars[x_start + t - 1];
            const AD<double> y0 = vars[y_start + t - 1];
            const AD<double> psi0 = vars[psi_start + t - 1];
            const AD<double> v0 = vars[v_start + t - 1];
            const AD<double> cte0 = vars[cte_start + t - 1];
            const AD<double> epsi0 = vars[epsi_start + t - 1];

            // Only consider the actuation at time t.
            const AD<double> delta0 = vars[delta_start + t - 1];
            const AD<double> a0 = vars[a_start + t - 1];

            const AD<double> x0_2 = x0 * x0;
            const AD<double> x0_3 = x0_2 * x0;
            const AD<double> f0 = coeffs[0] + coeffs[1] * x0 + coeffs[2]*x0_2 + coeffs[3]*x0_3; // evaluate the 3rd order poly
            const AD<double> psides0 = CppAD::atan(coeffs[1] + 2*coeffs[2]*x0 + 3*coeffs[3]*x0_2); // evaluate the derivative of the 3rd order poly

            // Here's `x` to get you started.
            // The idea here is to constraint this value to be 0.
            //
            // Recall the equations for the model:
            // x_[t] = x[t-1] + v[t-1] * cos(psi[t-1]) * dt
            // y_[t] = y[t-1] + v[t-1] * sin(psi[t-1]) * dt
            // psi_[t] = psi[t-1] + v[t-1] / Lf * delta[t-1] * dt
            // v_[t] = v[t-1] + a[t-1] * dt
            // cte[t] = f(x[t-1]) - y[t-1] + v[t-1] * sin(epsi[t-1]) * dt
            // epsi[t] = psi[t] - psides[t-1] + v[t-1] * delta[t-1] / Lf * dt
            fg[1 + x_start + t] = x1 - (x0 + v0 * CppAD::cos(psi0) * dt);
            fg[1 + y_start + t] = y1 - (y0 + v0 * CppAD::sin(psi0) * dt);
            fg[1 + psi_start + t] = psi1 - (psi0 + v0 * delta0 / Lf * dt);
            fg[1 + v_start + t] = v1 - (v0 + a0 * dt);
            fg[1 + cte_start + t] =
                    cte1 - ((f0 - y0) + (v0 * CppAD::sin(epsi0) * dt));
            fg[1 + epsi_start + t] =
                    epsi1 - ((psi0 - psides0) + v0 * delta0 / Lf * dt);
        }
    }
};

//
// MPC class definition implementation.
//
MPC::MPC() {}
MPC::~MPC() {}

Solution MPC::Solve(const Eigen::VectorXd &state, const Eigen::VectorXd &coeffs) {
    bool ok = true;
    typedef CPPAD_TESTVECTOR(double) Dvector;

    // Set the number of model variables (includes both states and inputs).
    // For example: If the state is a 4 element vector, the actuators is a 2
    // element vector and there are 10 timesteps. The number of variables is:
    //
    // 4 * 10 + 2 * 9
    const size_t n_vars = N * 6 + (N - 1) * 2;
    // Set the number of constraints
    const size_t n_constraints = N * 6;

    // Get references to the initial state values
    const double x = state[0];
    const double y = state[1];
    const double psi = state[2];
    const double v = state[3];
    const double cte = state[4];
    const double epsi = state[5];

    // Initialize each variable.
    Dvector vars(n_vars);
    for (size_t i = 0; i < n_vars; ++i) { // by default are all zero
        vars[i] = 0;
    }
    vars[x_start] = x;
    vars[y_start] = y;
    vars[psi_start] = psi;
    vars[v_start] = v;
    vars[cte_start] = cte;
    vars[epsi_start] = epsi;

    // Set lower and upper limits for each variable
    Dvector vars_lowerbound(n_vars);
    Dvector vars_upperbound(n_vars);

    // Set upper and lower limits for each state variable
    for (size_t i = 0; i < delta_start; ++i) {
        vars_lowerbound[i] = std::numeric_limits<double>::lowest();
        vars_upperbound[i] = std::numeric_limits<double>::max();
    }

    // Set upper and lower limits fo steer to -25 and 25 degrees in radians
    for (size_t i = delta_start; i < a_start; ++i) {
        vars_lowerbound[i] = deg2rad(-25);
        vars_upperbound[i] = deg2rad(25);
    }

    // Due to the latency in executing the steer, the lower and upper bounds must stick to the previous control for the entire latency time
    // Set the lower and upper bounds to the previous steer for the duration of the latency time
    for (size_t i = delta_start; i < delta_start + latency_dt_units; ++i) {
        vars_lowerbound[i] = delta_prev;
        vars_upperbound[i] = delta_prev;
    }

    // Set upper and lower limits for acceleration and decceleration
    for (size_t i = a_start; i < n_vars; ++i) {
        vars_lowerbound[i] = -1.0;
        vars_upperbound[i] =  1.0;
    }

    // Due to the latency in executing the throttle, the lower and upper bounds must stick to the previous control for the entire latency time
    // Set the lower and upper bounds to the previous throttle for the duration of the latency time
    for (size_t i = a_start; i < a_start+latency_dt_units; ++i) {
        vars_lowerbound[i] = a_prev;
        vars_upperbound[i] = a_prev;
    }

    // Set lower and upper limits for each constraint
    Dvector constraints_lowerbound(n_constraints);
    Dvector constraints_upperbound(n_constraints);

    // Initialize all constraints to 0
    for (size_t i = 0; i < n_constraints; ++i) {
        constraints_lowerbound[i] = 0;
        constraints_upperbound[i] = 0;
    }

    // Set both upper and lower constraint limits for the inital state to the inital state values
    constraints_lowerbound[x_start] = x;
    constraints_lowerbound[y_start] = y;
    constraints_lowerbound[psi_start] = psi;
    constraints_lowerbound[v_start] = v;
    constraints_lowerbound[cte_start] = cte;
    constraints_lowerbound[epsi_start] = epsi;
    constraints_upperbound[x_start] = x;
    constraints_upperbound[y_start] = y;
    constraints_upperbound[psi_start] = psi;
    constraints_upperbound[v_start] = v;
    constraints_upperbound[cte_start] = cte;
    constraints_upperbound[epsi_start] = epsi;

    // Object that computes objective and constraints
    FG_eval fg_eval(coeffs);

    //
    // NOTE: You don't have to worry about these options
    //
    // options for IPOPT solver
    std::string options;
    // Uncomment this if you'd like more print information
    options += "Integer print_level  0\n";
    // NOTE: Setting sparse to true allows the solver to take advantage
    // of sparse routines, this makes the computation MUCH FASTER. If you
    // can uncomment 1 of these and see if it makes a difference or not but
    // if you uncomment both the computation time should go up in orders of
    // magnitude.
    options += "Sparse  true        forward\n";
    options += "Sparse  true        reverse\n";
    // NOTE: Currently the solver has a maximum time limit of 0.5 seconds.
    // Change this as you see fit.
    options += "Numeric max_cpu_time          0.5\n";

    // place to return solution
    CppAD::ipopt::solve_result<Dvector> solution;

    // solve the problem
    CppAD::ipopt::solve<Dvector, FG_eval>(
                options, vars, vars_lowerbound, vars_upperbound, constraints_lowerbound,
                constraints_upperbound, fg_eval, solution);

    // Check some of the solution values
    ok &= solution.status == CppAD::ipopt::solve_result<Dvector>::success;
    cout << "ok:" << ok << endl;

    // Cost
    const double cost = solution.obj_value;
    std::cout << "Cost " << cost << std::endl;

    Solution sol;
    for (int i = 0; i < N-1 ; ++i) {
        cout << i << ": " << "solution.x[x_start+i]: " << solution.x[x_start+i] << "solution.x[y_start+i]: " << solution.x[y_start+i] << endl;
        sol.X.push_back(solution.x[x_start+i]);
        sol.Y.push_back(solution.x[y_start+i]);
        sol.Delta.push_back(solution.x[delta_start+i]);
        sol.A.push_back(solution.x[a_start+i]);
    }

    return sol;
}
