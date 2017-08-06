#include "PID.h"

using namespace std;

/*
* TODO: Complete the PID class.
*/

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp, double Ki, double Kd) {
    this->Kp = Kp;
    this->Ki = Ki;
    this->Kd = Kd;
}

void PID::UpdateError(double cte) {
    d_error = cte - p_error; // crt prop error -  prev prop error
    p_error = cte; // update crt proportional error
    i_error += cte; // update crt integral error (sum of error of the whole time interval)
}

double PID::TotalError() {
    return (-Kp * p_error - Kd * d_error - Ki * i_error);
}

