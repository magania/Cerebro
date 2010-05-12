/*
 * RBMLayer.cpp
 *
 *  Created on: May 7, 2010
 *      Author: magania
 */

#include <RBMLayer.hpp>

RBMLayer::RBMLayer(DataSet *data, int hidden_neurons) :
    _aleatorio(rng, uniform_number),
    _data(data),
    _vNeurons(data->dim()),
    _hNeurons(hidden_neurons)
    {
	std::cout << "RBMLayer ... " << std::endl;

	_W = new float*[_vNeurons];
	_W0 = new float*[_vNeurons];
	_W1 = new float*[_vNeurons];
	for (int i = 0; i < _vNeurons; i++) {
		_W[i] = new float[_hNeurons];
		_W0[i] = new float[_hNeurons];
		_W1[i] = new float[_hNeurons];
	}

	_vBias = new float[_vNeurons];
	_vBias0 = new float[_vNeurons];
	_vBias1 = new float[_vNeurons];

	_hBias = new float[_hNeurons];
	_hBias0 = new float[_hNeurons];
	_hBias1 = new float[_hNeurons];

	for (int v=0; v<_vNeurons; v++)
		for (int h=0; h<_hNeurons; h++)
			_W[v][h] = _W0[v][h] = _W1[v][h]= 0;

	for (int v=0; v<_vNeurons; v++)
		_vBias[v] = _vBias0[v] = _vBias1[v] = 0;

	for (int h=0; h<_hNeurons; h++)
		_hBias[h] = _hBias0[h] = _hBias1[h] = 0;
}

void RBMLayer::train(int batches, float epsilon, int cores) {
	std::cout << "RBMLayer train: cores " << cores << " epsilon " << epsilon << std::endl;
	_batches = batches;
	_epsilon = epsilon;
	_cores_ready = 0;
	_cores = cores;

	_hP = new float*[_cores];
    _vP = new float*[_cores];
	for (int i=0; i<_cores; i++){
		_hP[i] = new float[_hNeurons];
		_vP[i] = new float[_vNeurons];
	}

	boost::thread* threads[_cores];
	for (int i = 1; i < _cores; i++)
		threads[i] = new boost::thread(boost::ref(*this), i);

	operator()(0);

	for (int v=0; v<_vNeurons; v++)
		for (int h=0; h<_hNeurons; h++)
			std::cout << _W[v][h] << ' ';
	std::cout << std::endl;

	for (int i=0; i<_cores; i++){
		delete _hP[i];
		delete _vP[i];
	}
	delete _hP;
	delete _vP;

/*	int x=0;
	while (1) {
		x++;
	}
*/
//	boost::thread mythread(*this,1);
}

void RBMLayer::up(int core, float *V){
	for (int h=0; h<_hNeurons; h++){
		float sum=0;
		for (int v=0; v<_vNeurons; v++)
			sum += _W[v][h] * V[v];
		_hP[core][h] = 1.0/(1.0+exp(sum - _hBias[h]));
	}
}

void RBMLayer::up_d(int core, int sample){
//	std::cout << "Core " << core << " up_d" << std::endl;
	up(core, _data->get(sample));
}

void RBMLayer::up(int core){
//	std::cout << "Core " << core << " up" << std::endl;
	up(core, _vP[core]);
}

float RBMLayer::sample(float x){
	return x > _aleatorio()	;
}

void RBMLayer::down(int core){
//	std::cout << "Core " << core << " down" << std::endl;
	for (int v=0; v<_vNeurons; v++){
		float sum=0;
		for (int h=0; h<_hNeurons; h++)
			sum += _W[v][h] * sample(_hP[core][h]);
		_vP[core][v] = 1.0/(1.0+exp(sum - _vBias[v]));
	}
}

void RBMLayer::update_W0(int core, int sample){
	boost::mutex::scoped_lock  lock(_mutex_W0);
//	std::cout << "Core " << core << " update W0" << std::endl;
	for(int v=0; v<_vNeurons; v++)
		for(int h=0; h<_hNeurons; h++)
			_W0[v][h] += _data->get(sample)[v] * _hP[core][h];
}

void RBMLayer::update_vBias0(int core, int sample){
	boost::mutex::scoped_lock  lock(_mutex_vBias0);
//	std::cout << "Core " << core << " update vBias0" << std::endl;
	for(int v=0; v<_vNeurons; v++)
		_vBias0[v] += _data->get(sample)[v];
}

void RBMLayer::update_hBias0(int core){
	boost::mutex::scoped_lock  lock(_mutex_hBias0);
//	std::cout << "Core " << core << " update hBias0" << std::endl;
	for(int h=0; h<_hNeurons; h++)
		_hBias0[h] += _hP[core][h];
}

void RBMLayer::update_W1(int core){
	boost::mutex::scoped_lock  lock(_mutex_W1);
//	std::cout << "Core " << core << " update W1" << std::endl;
	for(int v=0; v<_vNeurons; v++)
		for(int h=0; h<_hNeurons; h++)
			_W1[v][h] += _vP[core][v] * _hP[core][h];
}

void RBMLayer::update_vBias1(int core){
	boost::mutex::scoped_lock  lock(_mutex_vBias1);
//	std::cout << "Core " << core << " update vBias1" << std::endl;
	for(int v=0; v<_vNeurons; v++)
		_vBias1[v] += _vP[core][v];
}

void RBMLayer::update_hBias1(int core){
	boost::mutex::scoped_lock  lock(_mutex_hBias1);
//	std::cout << "Core " << core << " update hBias1" << std::endl;
	for(int h=0; h<_hNeurons; h++)
		_hBias1[h] += _hP[core][h];
}

void RBMLayer::update1(int core, int x_size, float *x, float *x0, float *x1) {
	for (int i = core; i < x_size; i += _cores)
		x[i] += _epsilon * (x0[i] - x1[i]);
}

void RBMLayer::update2(int core, int x_size, int y_size, float **x, float **x0, float **x1) {
	for (int i = core; i < x_size; i += _cores)
		for (int j = 0; j < y_size; j++)
			x[i][j] += _epsilon * (x0[i][j] - x1[i][j]);
}

void RBMLayer::update_W(int core) {
	update2(core, _vNeurons, _hNeurons, _W, _W0, _W1);
}

void RBMLayer::update_vBias(int core) {
	update1(core, _vNeurons, _vBias, _vBias0, _vBias1);
}

void RBMLayer::update_hBias(int core) {
	update1(core, _hNeurons, _hBias, _hBias0, _hBias1);
}

void RBMLayer::synchronize() {
	std::cout << "synchonizing ..." << std::endl;
	boost::mutex::scoped_lock  lock(_mutex_ready);
	if (++_cores_ready == _cores){
		_cores_ready = 0;
		_cond_ready.notify_all();
	} else {
		_cond_ready.wait(_mutex_ready);
	}
}

void RBMLayer::update_weights(int core) {
	synchronize();

	update_W(core);
	update_vBias(core);
	update_hBias(core);

	for (int v=core; v<_vNeurons; v+=_cores)
		for (int h=0; h<_hNeurons; h++)
			_W0[v][h] = _W1[v][h] = 0;

	for (int v=core; v<_vNeurons; v+=_cores)
		_vBias0[v] = _vBias1[v] = 0;

	for (int h=core; h<_hNeurons; h+=_cores)
		_hBias0[h] = _hBias1[h] = 0;

	synchronize();
}

void RBMLayer::operator()(int core) {
	std::cout << "Core " << core << " Starting ... " << std::endl;
	for (int sample = core; sample<_data->size(); sample +=_cores){
		std::cout << core << ": Sample " << sample << std::endl;

		up_d(core,sample);

		update_W0(core,sample);
		update_vBias0(core,sample);
		update_hBias0(core);

		down(core);
		up(core);

		update_W1(core);
		update_vBias1(core);
		update_hBias1(core);


		if ( ( (sample % _batches) - core) == (_batches - _cores) || sample == (_data->size() - (_data->size() % _cores) + core - _cores) )
			update_weights(core);
	}
}
