/*
 * RBMLayer.hpp
 *
 *  Created on: May 7, 2010
 *      Author: magania
 */

#ifndef RBMLAYER_HPP_
#define RBMLAYER_HPP_

#include <boost/thread.hpp>

#include <boost/thread/mutex.hpp>
#include <boost/thread/condition.hpp>

#include <boost/random.hpp>
#include <ctime>

#include <xmmintrin.h>
#include <malloc.h>
#include <stdlib.h>

#include <iostream>
#include <DataSet.hpp>

class RBMLayer {
private:
	boost::mt19937 rng;
	boost::uniform_real<float> uniform_number;
	boost::variate_generator<boost::mt19937&, boost::uniform_real<float> > _aleatorio;

	int _vNeurons, _hNeurons;

	float **_W, **_W0, **_W1, *_vBias, *_vBias0, *_vBias1, *_hBias, *_hBias0, *_hBias1;
	float **_hP, **_vP;

	DataSet *__data;
	int __batch_size;
	int __cores, __cores_ready;
	float __epsilon;

	//Mutex to protect access
	boost::mutex _mutex_W0, _mutex_vBias0, _mutex_hBias0;
	boost::mutex _mutex_W1, _mutex_vBias1, _mutex_hBias1;

	//Condition to wait for all threads
	boost::mutex _mutex_ready;
	boost::condition _cond_ready;

	void up_d(int core, int sample);
	void up(int core);
	inline void up(int core, float *V);
	void down(int core);

	void update_W(int core);
	void update_vBias(int core);
	void update_hBias(int core);

	void update_W0(int core, int sample);
	void update_vBias0(int core, int sample);
	void update_hBias0(int core);

	void update_W1(int core);
	void update_vBias1(int core);
	void update_hBias1(int core);

	void up(int core, int visible_size, float *visible, int hidden_size, float *hidden);
	void update1(int core, int x_size, float *x, float *x0, float *x1);
	void update2(int core, int x_size, int y_size, float **x, float **x0, float **x1);

	void synchronize();
	void update_weights(int core);

	inline float sample(float x);
	inline void mul(int size, float* x, float* a, float k);
	inline void add(int size, float* a, float* b);
	inline void del(int size, float *x, float* a, float* b, float k);
	inline void zero(int size, float *x);

public:
	void operator()(int core);

	RBMLayer(int visible_neurons, int hidden_neurons);
	~RBMLayer();
	void train(DataSet& data, int batches, float epsilon, int cores = 1);
};

#endif /* RBMLAYER_HPP_ */
