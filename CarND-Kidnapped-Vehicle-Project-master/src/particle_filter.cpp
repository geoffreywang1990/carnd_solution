/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

// declare a random engine for being used by all methods
static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
    //   x, y, theta and their uncertainties from GPS) and all weights to 1.
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    // Set the number of particle
    num_particles = 20;
    particles.reserve(num_particles);

    // Creates 3 normal (Gaussian) distributions for x, y and theta
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    // Initialize all particles
    for (int p_i = 0; p_i < num_particles; ++p_i) {
        // set position to first position (based on estimates of x, y, theta and their uncertainties from GPS)
        double p_x, p_y, p_theta, p_weigth;
        p_x = dist_x(gen);
        p_y = dist_y(gen);
        p_theta = dist_theta(gen);
        // set weight to 1
        p_weigth = 1.0;
        // set all weights to 1
        Particle p {p_i, p_x, p_y, p_theta, p_weigth};
        // store particle
        particles.push_back(p);
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // TODO: Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/

    // Creates 3 normal (Gaussian) distributions for x, y and theta
    normal_distribution<double> dist_x(0.0, std_pos[0]); // x noise generator
    normal_distribution<double> dist_y(0.0, std_pos[1]); // y noise generator
    normal_distribution<double> dist_theta(0.0, std_pos[2]); // theta noise generator

    // Add measurements to each particle and add random Gaussian noise.
    for (int p_i = 0; p_i < num_particles; ++p_i) {
        Particle &p = particles[p_i];
        if (std::fabs(yaw_rate) > 0.001) {
            // use the equations for updating x, y and the yaw angle when the yaw rate is not equal to zero
            p.x += (velocity/yaw_rate) * (sin(p.theta + yaw_rate*delta_t) - sin(p.theta)) + dist_x(gen);
            p.y += (velocity/yaw_rate) * (cos(p.theta) - cos(p.theta + yaw_rate*delta_t)) + dist_y(gen);
        } else {
            p.x += velocity * delta_t * cos(p.theta) + dist_x(gen);
            p.y += velocity * delta_t * sin(p.theta) + dist_y(gen);
        }
        p.theta += yaw_rate*delta_t + dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
    //   implement this method and use it as a helper during the updateWeights phase.

    // mapping of predicted and observed landmarks with the shortest distance
    for (int o_i = 0; o_i < observations.size(); ++o_i) {

        // get current observation
        const LandmarkObs &o = observations[o_i];

        // landmark id of the map associated with the current observation
        int map_id = -1;

        // set the minimum distance to the maximum possible
        double min_distance = numeric_limits<double>::max();

        for (int p_i = 0; p_i < predicted.size(); ++p_i) {
            // get current prediction
            const LandmarkObs &p = predicted[p_i];

            // get the distance between current and predicted landmarks
            const double current_distance = dist(o.x, o.y, p.x, p.y);

            // find predicted and observed landmarks with the shortest distance
            if (current_distance < min_distance) {
                min_distance = current_distance;
                map_id = p.id;
            }
        }

        // set the id of the current observation to the nearest predicted map's landmark id
        observations[o_i].id = map_id;
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   std::vector<LandmarkObs> observations, Map map_landmarks) {
    // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
    //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation
    //   3.33
    //   http://planning.cs.uiuc.edu/node99.html

    // Update the weights of each particle using a mult-variate Gaussian distribution
    for (int p_i = 0; p_i < num_particles; ++p_i) {

        // get current particle
        Particle &p = particles[p_i];

        // keep only landmark locations in the nearby of the particle's location
        vector<LandmarkObs> predictions;
        for (int lm_i = 0; lm_i < map_landmarks.landmark_list.size(); ++lm_i) {
            // get current landmark
            Map::single_landmark_s &lm = map_landmarks.landmark_list[lm_i];
            // keep only landmarks within the sensor range of the particle
            if (fabs(lm.x_f - p.x) <= sensor_range && fabs(lm.y_f - p.y) <= sensor_range) {
                // add prediction to vector
                predictions.push_back( LandmarkObs{ lm.id_i, lm.x_f, lm.y_f } );
            }
        }

        // transform all observations from the vehicle coordinates to the map coordinates
        vector<LandmarkObs> observations_map;
        observations_map.reserve(observations.size());
        for (int o_i = 0; o_i < observations.size(); ++o_i) {
            const double t_x = p.x + cos(p.theta) * observations[o_i].x - sin(p.theta) * observations[o_i].y;
            const double t_y = p.y + sin(p.theta) * observations[o_i].x + cos(p.theta) * observations[o_i].y;
            observations_map.push_back( LandmarkObs{ observations[o_i].id, t_x, t_y } );
        }

        // find the associations between predictions and transformed observations for the current particle
        dataAssociation(predictions, observations_map);

        // reset the weight
        particles[p_i].weight = 1.0;

        for (int o_i = 0; o_i < observations_map.size(); ++o_i) {

            const double o_x = observations_map[o_i].x;
            const double o_y = observations_map[o_i].y;
            const int o_id = observations_map[o_i].id;

            // get the x,y coordinates of the prediction associated with the current observation
            double pred_x, pred_y;
            for (int pred_i = 0; pred_i < predictions.size(); ++pred_i) {
                if (predictions[pred_i].id == o_id) {
                    pred_x = predictions[pred_i].x;
                    pred_y = predictions[pred_i].y;
                }
            }

            // compute the new weight of this observation using a multivariate Gaussian
            const double std_x = std_landmark[0];
            const double std_y = std_landmark[1];
            const double obs_w = ( 1. / (2.*M_PI*std_x*std_y)) * exp( -( pow(pred_x-o_x,2.) / (2.*pow(std_x, 2.)) + pow(pred_y-o_y,2.) / (2.*pow(std_y, 2.)) ) );
            particles[p_i].weight *= obs_w;
        }
    }
}

void ParticleFilter::resample() {
    // TODO: Resample particles with replacement with probability proportional to their weight.
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    // declare the new set of particles after current resampling
    vector<Particle> resampled_particles;
    resampled_particles.reserve(num_particles);

    // get the max weight of current particles
    double max_weight = 0.0;
    for(int p_i = 0; p_i < num_particles; ++p_i) {
        max_weight = max(max_weight, particles[p_i].weight);
    }

    // generate a random starting index for the resampling wheel
    uniform_int_distribution<int> uni_int_dist(0, num_particles-1);

    // declare a uniform random distribution with the range: [0.0, max_weight)
    uniform_real_distribution<double> uni_real_dist(0.0, max_weight);

    // spin the resampling wheel
    int index = uni_int_dist(gen);
    double beta = 0.0;
    for (int i = 0; i < num_particles; i++) {
        beta += uni_real_dist(gen) * 2.0 * max_weight;
        while (beta > particles[index].weight) {
            beta -= particles[index].weight;
            index = (index + 1) % num_particles;
        }
        resampled_particles.push_back(particles[index]);
    }

    particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    //Clear the previous associations
    particle.associations.clear();
    particle.sense_x.clear();
    particle.sense_y.clear();

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
    vector<int> v = best.associations;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
    vector<double> v = best.sense_x;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
    vector<double> v = best.sense_y;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
