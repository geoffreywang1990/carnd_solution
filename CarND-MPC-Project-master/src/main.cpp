#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "MPC.h"
#include "json.hpp"

// for convenience
using json = nlohmann::json;
using namespace std;

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
    auto found_null = s.find("null");
    auto b1 = s.find_first_of("[");
    auto b2 = s.rfind("}]");
    if (found_null != string::npos) {
        return "";
    } else if (b1 != string::npos && b2 != string::npos) {
        return s.substr(b1, b2 - b1 + 2);
    }
    return "";
}

// Evaluate a polynomial.
inline double polyeval(const Eigen::VectorXd &coeffs, const double x) {
    double result = 0.0;
    for (int i = 0; i < coeffs.size(); ++i) {
        result += coeffs[i] * pow(x, i);
    }
    return result;
}

// Fit a polynomial.
// Adapted from
// https://github.com/JuliaMath/Polynomials.jl/blob/master/src/Polynomials.jl#L676-L716
Eigen::VectorXd polyfit(const Eigen::VectorXd &xvals, const Eigen::VectorXd &yvals,
                        const int order) {
    assert(xvals.size() == yvals.size());
    assert(order >= 1 && order <= xvals.size() - 1);
    Eigen::MatrixXd A(xvals.size(), order + 1);

    for (int i = 0; i < xvals.size(); i++) {
        A(i, 0) = 1.0;
    }

    for (int j = 0; j < xvals.size(); j++) {
        for (int i = 0; i < order; i++) {
            A(j, i + 1) = A(j, i) * xvals(j);
        }
    }

    auto Q = A.householderQr();
    auto result = Q.solve(yvals);
    return result;
}

// The cross-track error cte of the vehicle in the vehicle coordinate is when x=0
inline double computeCte(const Eigen::VectorXd &coeffs) {
    return polyeval(coeffs, 0);
}

// The orientation error epsi can be calculated as the tangential angle of the polynomial f evaluated
// at x​t as: arctan(f'​​(x​_t​​)), where f'​​ is the derivative of the polynomial.
// After derivation the f becomes: c1 + c2 * x + c3 * x^2, and for x=0, for vehicle coordinate, it reduces to: -atan (c1)
inline double computeEpsi(const Eigen::VectorXd &coeffs) {
    return -atan(coeffs[1]);
}

// It transforms a trajectory (a list of 2D points) from the map coordinate to the vechile coordinate system
inline void transformFromMapToVechicleCoordinate(const double x, const double y, const double psi,
                                       const vector<double> & ptsx, const vector<double> & ptsy,
                                       Eigen::VectorXd & ptsx_vechicle, Eigen::VectorXd & ptsy_vechicle) {
    assert(ptsx.size() == ptsy.size());
    const size_t n = ptsx.size();
    ptsx_vechicle.resize(n);
    ptsy_vechicle.resize(n);
    // rotate each point clockwise by 90 deg, (−90°)
    for (size_t i=0; i<ptsx.size() ; ++i) {
        ptsx_vechicle[i] = cos(psi) * (ptsx[i] - x) + sin(psi) * (ptsy[i] - y);
        ptsy_vechicle[i] = -sin(psi) * (ptsx[i] - x) + cos(psi) * (ptsy[i] - y);
    }
}

int main() {
    uWS::Hub h;

    // MPC is initialized here!
    MPC mpc;

    h.onMessage([&mpc](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                uWS::OpCode opCode) {
        // "42" at the start of the message means there's a websocket message event.
        // The 4 signifies a websocket message
        // The 2 signifies a websocket event
        string sdata = string(data).substr(0, length);
        cout << sdata << endl;
        if (sdata.size() > 2 && sdata[0] == '4' && sdata[1] == '2') {
            string s = hasData(sdata);
            if (s != "") {
                auto j = json::parse(s);
                string event = j[0].get<string>();
                if (event == "telemetry") {
                    // j[1] is the data JSON object
                    const vector<double> ptsx = j[1]["ptsx"];
                    const vector<double> ptsy = j[1]["ptsy"];
                    const double px = j[1]["x"];
                    const double py = j[1]["y"];
                    const double psi = j[1]["psi"];
                    const double v = j[1]["speed"];

                    /*
                     * TODO: Calculate steering angle and throttle using MPC.
                     *
                     * Both are in between [-1, 1].
                     *
                     */

                    // Transform the trajectory which the vechicle must follow from the map coordinate system to vechicle coordinate system.
                    Eigen::VectorXd ptsx_vechicle, ptsy_vechicle;
                    transformFromMapToVechicleCoordinate(px, py, psi, ptsx, ptsy, ptsx_vechicle, ptsy_vechicle);

                    // use the reference trajectory points in the vechicle coordinate system to  to fit a 3rd order polynomial
                    const Eigen::VectorXd refPolyCoeffs = polyfit(ptsx_vechicle, ptsy_vechicle, 3);

                    // compute the cross-track error using the current 3rd order polynomial
                    const double cte = computeCte(refPolyCoeffs);

                    // compute the orientation error using the current 3rd order polynomial
                    const double epsi = computeEpsi(refPolyCoeffs);

                    // initialize the state to be used by the MPS as starting point for the optimization
                    // in the vehicle coordinate system the state (x,y, psi) is zero
                    Eigen::VectorXd state(6);
                    state << 0, 0, 0, v, cte, epsi;

                    // compute an estimation of the optimal trajectory
                    // the solution contains the current and future values
                    const Solution solution = mpc.Solve(state, refPolyCoeffs);

                    // use the steer and throttle in the future, where the latency would be eliminated
                    // the for a dt=0.05 and latency=0.1, the value that eliminates the latency is 2 positions after the first current value
                    const double steer_value = solution.Delta[latency_dt_units];
                    const double throttle_value = solution.A[latency_dt_units];
                    mpc.delta_prev = steer_value;
                    mpc.a_prev = throttle_value;

                    cout << "px:"<< px <<", py:"<< py <<", psi:"<< psi <<", v:"<< v
                         << ", cte:"<< cte <<", epsi:"<< epsi
                         << ", steer_value:"<< steer_value <<", throttle:"<< throttle_value <<endl;

                    json msgJson;
                    // NOTE: Remember to divide by deg2rad(25) before you send the steering value back.
                    // Otherwise the values will be in between [-deg2rad(25), deg2rad(25] instead of [-1, 1].
                    // in the simulator positive angles are negative
                    msgJson["steering_angle"] = -1 * steer_value / deg2rad(25);
                    msgJson["throttle"] = throttle_value;

                    //Display the MPC predicted trajectory
                    msgJson["mpc_x"] = solution.X;
                    msgJson["mpc_y"] = solution.Y;

                    //Display the reference line
                    vector<double> next_x_vals;
                    vector<double> next_y_vals;

                    //.. add (x,y) points to list here, points are in reference to the vehicle's coordinate system
                    // the points in the simulator are connected by a Yellow line
                    for (unsigned i=0 ; i < ptsx.size(); ++i) {
                        next_x_vals.push_back(ptsx_vechicle(i));
                        next_y_vals.push_back(ptsy_vechicle(i));
                    }
                    msgJson["next_x"] = next_x_vals;
                    msgJson["next_y"] = next_y_vals;

                    auto msg = "42[\"steer\"," + msgJson.dump() + "]";
                    std::cout << msg << std::endl;
                    // Latency
                    // The purpose is to mimic real driving conditions where
                    // the car does actuate the commands instantly.
                    //
                    // Feel free to play around with this value but should be to drive
                    // around the track with 100ms latency.
                    //
                    // NOTE: REMEMBER TO SET THIS TO 100 MILLISECONDS BEFORE
                    // SUBMITTING.
                    this_thread::sleep_for(chrono::milliseconds(100));
                    ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
                }
            } else {
                // Manual driving
                std::string msg = "42[\"manual\",{}]";
                ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
            }
        }
    });

    // We don't need this since we're not using HTTP but if it's removed the
    // program
    // doesn't compile :-(
    h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data,
                    size_t, size_t) {
        const std::string s = "<h1>Hello world!</h1>";
        if (req.getUrl().valueLength == 1) {
            res->end(s.data(), s.length());
        } else {
            // i guess this should be done more gracefully?
            res->end(nullptr, 0);
        }
    });

    h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
        std::cout << "Connected!!!" << std::endl;
    });

    h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                      char *message, size_t length) {
        ws.close();
        std::cout << "Disconnected" << std::endl;
    });

    int port = 4567;
    if (h.listen(port)) {
        std::cout << "Listening to port " << port << std::endl;
    } else {
        std::cerr << "Failed to listen to port" << std::endl;
        return -1;
    }
    h.run();
}
