/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 *      Author: Thomas Weustenfeld
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
// added by TW:
#include <chrono>
#include <ctime>

#include "particle_filter.h"

using namespace std;

static bool print_debug = false;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	num_particles = 10;

	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	default_random_engine gen(seed);

	// Set standard deviations for x, y, and theta.
	double std_x, std_y, std_theta;
	std_x     = std[0];
	std_y     = std[1];
	std_theta = std[2];
	
	// This line creates a normal (Gaussian) distribution for x, y and theta
	normal_distribution<double> dist_x(x, std_x);
	normal_distribution<double> dist_y(y, std_y);
	normal_distribution<double> dist_theta(theta, std_theta);

	// initialize weights to 1
	weights.resize(num_particles, 0);
	std::fill(weights.begin(), weights.end(), 1.0f);

	particles.clear();
	for( unsigned int i = 0; i < num_particles; ++i ) {

		// create new particle
		Particle new_particle;		
		new_particle.id    = i + 1; 

		// Sample  and from these normal distrubtions
		// where "gen" is the random engine initialized earlier (line 21).
		new_particle.x     = dist_x(gen);
		new_particle.y     = dist_y(gen);
		new_particle.theta = dist_theta(gen);

		// set weight to 1
		new_particle.weight = 1.0f;

		// Print your samples to the terminal.
		if( print_debug ) cout << "Sample " << new_particle.id << " " << new_particle.x << " " << new_particle.y << " " << new_particle.theta << endl;

		// append new particle
		particles.push_back(new_particle);
	}
    if( print_debug ) cout << "WEIGHTS = " << weights[0] << endl;

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	default_random_engine gen(seed);

	// Set standard deviations for x, y, and theta.
	double std_x, std_y, std_theta; 
	std_x     = std_pos[0];
	std_y     = std_pos[1];
	std_theta = std_pos[2];
	
	// This line creates a normal (Gaussian) distribution for x, y and theta
	normal_distribution<double> dist_x(0, std_x);
	normal_distribution<double> dist_y(0, std_y);
	normal_distribution<double> dist_theta(0, std_theta);

    if( print_debug ) cout << "PREDICT:" << endl;
    for(unsigned int i = 0; i < num_particles; ++i) {

        // measurement noise
        double x_noise     = dist_x(gen);
        double y_noise     = dist_y(gen);
		double theta_noise = dist_theta(gen);

		// old and new position
		double x_cur, y_cur, theta_cur;
		double x_new, y_new, theta_new;

		x_cur = particles[i].x;
		y_cur = particles[i].y;
		theta_cur = particles[i].theta;

		if (fabs(yaw_rate) < 0.001) {
			// Avoid division by zero
			theta_new = theta_cur;
			x_new = x_cur + velocity * std::cos( theta_cur ) * delta_t;
			y_new = y_cur + velocity * std::sin( theta_cur ) * delta_t;
		} else {
			// Update particles according to motion model
			theta_new = theta_cur + yaw_rate * delta_t;
			x_new = x_cur + ( velocity / yaw_rate ) * ( std::sin( theta_new ) - std::sin( theta_cur ) );
			y_new = y_cur + ( velocity / yaw_rate ) * ( std::cos( theta_cur ) - std::cos( theta_new ) );
		}

		particles[i].x = x_new + x_noise;
		particles[i].y = y_new + y_noise;
		particles[i].theta = theta_new + theta_noise;
        if( print_debug ) cout << " Sample " << particles[i].id << " " << particles[i].x << " " << particles[i].y << " " << particles[i].theta << endl;
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	// TODO!!
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	// helper for calculating norm distribution
	//definition of one over square root of 2*pi:
	struct helper {
		//definition square:
		static float squared(float x) {
			return x*x;
		}
     	// normpdf(X,mu,sigma) computes the probability function at values x using the
     	// normal distribution with mean mu and standard deviation std. x, mue and
     	// sigma must be scalar! The parameter std must be positive.
		static float normpdf(float x, float mu, float std) {
			const float ONE_OVER_SQRT_2PI = 1/sqrt(2*M_PI) ;
			return (ONE_OVER_SQRT_2PI/std)*exp(-0.5*squared((x-mu)/std));
		}
		// 2d normal distribution assuming no correlation of x and y
		static float normpdf2(float x, float y, float mu_x, float mu_y, float std_x, float std_y) {
			const float ONE_OVER_SQRT_2PI = 1/sqrt(2*M_PI) ;
			return (ONE_OVER_SQRT_2PI/(std_x*std_y))*exp(-0.5*( squared((x-mu_x)/std_x) + squared((y-mu_y)/std_y) ));
		}
	};

    for( unsigned int i = 0; i < num_particles; ++i ) {

		// Transform observations from vehicle to map
		std::vector<LandmarkObs> observations_map;
        observations_map.clear();

		for( unsigned int j = 0; j < observations.size(); ++j )
		{
			double x_obs, y_obs;
			x_obs = observations[j].x;
			y_obs = observations[j].y;

			double x_veh, y_veh, theta_veh;
			x_veh     = particles[i].x;
			y_veh     = particles[i].y;
			theta_veh = particles[i].theta;

			double x_map, y_map;
			x_map = x_obs * std::cos(theta_veh) - y_obs * std::sin(theta_veh) + x_veh;
			y_map = x_obs * std::sin(theta_veh) + y_obs * std::cos(theta_veh) + y_veh;

			LandmarkObs obs_map;
			obs_map.x = x_map;
			obs_map.y = y_map;
			observations_map.push_back(obs_map);
		}

		particles[i].weight = 1.0f;

		for( unsigned int j = 0; j < observations_map.size(); ++j)
		{
			double closest_dist = sensor_range;
			Map::single_landmark_s* closest_lm = NULL;

			for( unsigned int k = 0; k < map_landmarks.landmark_list.size(); ++k )
			{
				double lm_x, lm_y;
				lm_x = map_landmarks.landmark_list[k].x_f;
				lm_y = map_landmarks.landmark_list[k].y_f;

				double cur_dist;
				cur_dist = dist(observations_map[j].x, observations_map[j].y, lm_x, lm_y);
				
				if( cur_dist < closest_dist ) {
					closest_dist = cur_dist;
					closest_lm = &map_landmarks.landmark_list[k];
				}
			}

			if( closest_lm ) {
				double lm_x, lm_y;
				lm_x = closest_lm->x_f;
				lm_y = closest_lm->y_f;

/*				double x_part, y_part, theta_part;
				x_part     = particles[i].x;
				y_part     = particles[i].y;
                theta_part = particles[i].theta;*/

                double x_obs, y_obs;
                x_obs = observations_map[j].x;
                y_obs = observations_map[j].y;
				
				double std_x, std_y;
				std_x = std_landmark[0];
				std_y = std_landmark[1];
				
				double meas_prob;
//				meas_prob = helper::normpdf2(x_part, y_part, lm_x, lm_y, std_x, std_y);
//                meas_prob = 1/(2.0*M_PI*std_x*std_y)*std::exp(-(std::pow(x_part-lm_x,2.0)/(2.0*pow(std_x,2.0))+pow(y_part-lm_y,2.0)/(2.0*pow(std_y,2.0))));
                meas_prob = 1/(2.0*M_PI*std_x*std_y)*std::exp(-(std::pow(x_obs-lm_x,2.0)/(2.0*pow(std_x,2.0))+pow(y_obs-lm_y,2.0)/(2.0*pow(std_y,2.0))));
				if( print_debug ) 		cout << "  obs(" << x_obs <<  "," << y_obs << " ), lm(" << lm_x << "," << lm_y << "), std(" << std_x << "," << std_y << ") --> p[" << j << "] =  " << meas_prob << endl;
				particles[i].weight *= meas_prob;
			} else {
				cout << "no close landmark" << endl;
			}
		}
		weights[i] = particles[i].weight;
    }

    if( print_debug ) for( unsigned int i = 0; i < num_particles; ++i ) {
		cout << "  w[" << i << "]: " << weights[i] << endl;
	}
}

void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	// see: https://stackoverflow.com/questions/26475595/pseudo-random-number-generator-gives-same-first-output-but-then-behaves-as-expec
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	default_random_engine gen(seed);
	uniform_real_distribution<double> beta_uniform(0.0, 1.0);
	uniform_int_distribution<int> init_idx(0, num_particles-1);

	double beta, w_max;
	unsigned int idx;
	idx = init_idx(gen);
	beta = 0.0;
	w_max = *std::max_element(weights.begin(), weights.end());
    if( print_debug ) {
        cout << "idx: " << idx << endl;
        cout << "w_max: " << w_max << endl;
    }

	std::vector<Particle> new_particles;
	for( unsigned int i = 0; i < num_particles; ++i ) {
		beta += beta_uniform(gen) * 2.0 * w_max;
		while( beta > weights[idx] ) {
			beta -= weights[idx];
			idx = ( idx + 1 ) % num_particles;
		}
		new_particles.push_back( particles[idx] );
	}
	particles = new_particles;

    if( print_debug ) { cout << "RESAMPLE:" << endl;
        for( unsigned int i = 0; i < num_particles; ++i ) {
            cout << " Sample " << particles[i].id << " " << particles[i].x << " " << particles[i].y << " " << particles[i].theta << endl;
        }
    }
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
