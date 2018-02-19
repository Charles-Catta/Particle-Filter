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
#define EPSILON 0.0001

using namespace std;
default_random_engine generator;


void ParticleFilter::init(double x, double y, double theta, double std[]) {
	num_particles = 500;
	weights = vector<double>(num_particles);
	particles = vector<Particle>(num_particles);

	normal_distribution<double> N_theta_i(0, std[2]);
	normal_distribution<double> N_x_i(0, std[0]);
	normal_distribution<double> N_y_i(0, std[1]);

	for (int i = 0; i < num_particles; i++) {
		Particle p;
		p.x = x + N_x_i(generator);
		p.y = y + N_y_i(generator);
		p.theta = theta + N_theta_i(generator);
		p.weight = 1.0;
		p.id = i;
		particles[i] = p;
		weights[i] = p.weight;
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	normal_distribution<double> N_x(0, std_pos[0]);
	normal_distribution<double> N_y(0, std_pos[1]);
	normal_distribution<double> N_theta(0, std_pos[2]);

	for (int i = 0; i < num_particles; i++) {
		if (fabs(yaw_rate) < EPSILON) {
			particles[i].x += velocity * delta_t * cos(particles[i].theta) + N_x(generator);
			particles[i].y += velocity * delta_t * sin(particles[i].theta) + N_y(generator);
			particles[i].theta = N_theta(generator);
		} else {
			particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta)) + N_x(generator);
			particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t)) + N_y(generator);
			particles[i].theta += yaw_rate * delta_t + N_theta(generator);
		}
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	for (int obs_idx = 0; obs_idx < observations.size(); obs_idx++) {
		double closest_delta = numeric_limits<double>::max();
		int idx = 0;
		for (int pred_idx = 0; pred_idx < predicted.size(); pred_idx++) {
			double current_delta = dist(predicted[pred_idx].x, predicted[pred_idx].y, observations[obs_idx].x, observations[obs_idx].y);
			if (current_delta < closest_delta) {
				observations[obs_idx].id = idx;
				closest_delta = current_delta;
			}
			idx++;
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
			for (int prt_idx = 0; prt_idx < num_particles; prt_idx++) {
				vector<LandmarkObs> sensor_range_map;

				// Vector of landmarks within range
				for (int lmk_idx = 0; lmk_idx < map_landmarks.landmark_list.size(); lmk_idx++) {
					double lmk_x = map_landmarks.landmark_list[lmk_idx].x_f;
					double lmk_y = map_landmarks.landmark_list[lmk_idx].y_f;
					int lmk_id = map_landmarks.landmark_list[lmk_idx].id_i;
					double delta = dist(particles[prt_idx].x, particles[prt_idx].y, lmk_x, lmk_y);
					if (delta <= sensor_range) {
						LandmarkObs in_range_ob = {lmk_id, lmk_x, lmk_y};
						sensor_range_map.push_back(in_range_ob);
					}
				}

				vector<LandmarkObs> obs_map;
				// Vehicle coordinate system => World coordinate system
				for (int obs_idx = 0; obs_idx < observations.size(); obs_idx++) {
					double obs_x = particles[prt_idx].x + observations[obs_idx].x * cos(particles[prt_idx].theta) - observations[obs_idx].y * sin(particles[prt_idx].theta);
					double obs_y = particles[prt_idx].y + observations[obs_idx].x * sin(particles[prt_idx].theta) + observations[obs_idx].y * cos(particles[prt_idx].theta);
					LandmarkObs transformed_observation = {observations[obs_idx].id, obs_x, obs_y};
					obs_map.push_back(transformed_observation);
				} 

				if (sensor_range_map.size() == 0) {
            continue;
        }

				particles[prt_idx].weight = 1.0;
				dataAssociation(sensor_range_map, obs_map);

				double sig_sqrd_x = 2.0 * pow(std_landmark[0], 2);
				double sig_sqrd_y = 2.0 * pow(std_landmark[1], 2);
				double gauss_norm = 2.0 * M_PI * std_landmark[0] * std_landmark[1];
				
				for (int obs_idx = 0; obs_idx < obs_map.size(); obs_idx++) {
					double x_delta_mu = pow(obs_map[obs_idx].x - sensor_range_map[obs_map[obs_idx].id].x, 2);
					double y_delta_mu = pow(obs_map[obs_idx].y - sensor_range_map[obs_map[obs_idx].id].y, 2);
					particles[prt_idx].weight *= exp(-( x_delta_mu / sig_sqrd_x + y_delta_mu / sig_sqrd_y)) / gauss_norm;
				}
				weights[prt_idx] = particles[prt_idx].weight;
			} // end for
}

void ParticleFilter::resample() {
	vector<double> weights;
	for (int i = 0; i < particles.size(); i++) {
		weights.push_back(particles[i].weight);
	}
	discrete_distribution<int> distribution(weights.begin(), weights.end());

	vector<Particle> resampled_particles;
	resampled_particles.reserve(num_particles);

	for (int i = 0; i < num_particles; i++) {
		resampled_particles.push_back(particles[distribution(generator)]);
	}
	particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y) {
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

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
