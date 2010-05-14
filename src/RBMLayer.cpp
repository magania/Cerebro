/*
 * RBMLayer.cpp
 *
 *  Created on: May 7, 2010
 *      Author: magania
 */

#include <RBMLayer.hpp>

RBMLayer::RBMLayer(int visible_neurons, int hidden_neurons) :
    _aleatorio(rng, uniform_number),
    _vNeurons(visible_neurons),
    _hNeurons(hidden_neurons)
    {
	std::cout << "RBMLayer constructor ... " << std::endl;

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
			_W[v][h] = _aleatorio()*0.1;

	for (int v=0; v<_vNeurons; v++)
		for (int h=0; h<_hNeurons; h++)
			_W0[v][h] = _W1[v][h]= 0;

	for (int v=0; v<_vNeurons; v++)
		_vBias[v] = _vBias0[v] = _vBias1[v] = 0;

	for (int h=0; h<_hNeurons; h++)
		_hBias[h] = _hBias0[h] = _hBias1[h] = 0;
}

RBMLayer::~RBMLayer(){
	std::cout << "RBMLayer destructor ..." << std::endl;

	for (int i = 0; i < _vNeurons; i++) {
		delete _W[i];
		delete _W0[i];
		delete _W1[i];
	}
	delete _W;
	delete _W0;
	delete _W1;


	delete _vBias;
	delete _vBias0;
	delete _vBias1;

	delete _hBias;
	delete _hBias0;
	delete _hBias1;

}

void RBMLayer::train(DataSet *data, int batch_size, float epsilon, int cores) {
	std::cout << "RBMLayer train: cores " << cores << " epsilon " << epsilon << std::endl;
	__data = data;
	__batch_size = batch_size;
	__epsilon = epsilon;
	__cores_ready = 0;
	__cores = cores;

	_hP = new float*[__cores];
        _vP = new float*[__cores];
	for (int i=0; i<__cores; i++){
		_hP[i] = new float[_hNeurons];
		_vP[i] = new float[_vNeurons];
	}

	boost::thread* threads[__cores];
	for (int i = 0; i < __cores; i++)
		threads[i] = new boost::thread(boost::ref(*this), i);

	for (int i = 0; i < __cores; i++)
		threads[i]->join();

	for (int i = 0; i < __cores; i++)
		delete threads[i];


	for (int h=0; h<_hNeurons; h++)
		std:: cout << "Bias " << _hBias[h] << std::endl;

	DataSet xdata;
	xdata.size = _hNeurons;
	xdata.dim = _vNeurons;
	xdata.data = new float*[_hNeurons];
	for (int h=0; h<_hNeurons; h++)
		xdata.data[h] = new float [_vNeurons];

	float min = _W[0][0];
	float max = _W[0][0];
        for (int v=0; v<_vNeurons; v++)
                for (int h=0; h<_hNeurons; h++){
			if (_W[v][h] < min ) min =_W[v][h];
			if (_W[v][h] > max ) max =_W[v][h];
 		}
	std::cout << "Min: " << min  << " Max: " << max << std::endl;


	for (int v=0; v<_vNeurons; v++)
		for (int h=0; h<_hNeurons; h++)
			xdata.data[h][v] = (_W[v][h] - min)/(max-min);

	DataSet::print(&xdata, 0, 28);
	DataSet::print(&xdata, 1, 28);
	DataSet::print(&xdata, 2, 28);
	 for (int h=0; h<_hNeurons; h++)
                delete xdata.data[h];
        delete xdata.data;


	for (int i=0; i<__cores; i++){
		delete _hP[i];
		delete _vP[i];
	}
	delete _hP;
	delete _vP;

//	boost::thread mythread(*this,1);
}

inline void RBMLayer::up(int core, float *V){
	for (int h=0; h<_hNeurons; h++){
		float sum=0;
		for (int v=0; v<_vNeurons; v++)
			sum += _W[v][h] * V[v];
		_hP[core][h] = 1.0/(1.0+exp(-sum -_hBias[h]));
	}
}

void RBMLayer::up_d(int core, int sample){
//	std::cout << "Core " << core << " up_d" << std::endl;
	up(core, __data->data[sample]);
}

void RBMLayer::up(int core){
//	std::cout << "Core " << core << " up" << std::endl;
	up(core, _vP[core]);
}

inline float RBMLayer::sample(float x){
	return x > _aleatorio()	;
}

void RBMLayer::down(int core){
//	std::cout << "Core " << core << " down" << std::endl;
	for (int v=0; v<_vNeurons; v++){
		float sum=0;
		for (int h=0; h<_hNeurons; h++)
			sum += _W[v][h] * sample(_hP[core][h]);
		_vP[core][v] = 1.0/(1.0+exp(-sum -_vBias[v]));
	}
}

void RBMLayer::update_W0(int core, int sample){
	boost::mutex::scoped_lock  lock(_mutex_W0);
//	std::cout << "Core " << core << " update W0" << std::endl;
	for(int v=0; v<_vNeurons; v++)
		for(int h=0; h<_hNeurons; h++)
			_W0[v][h] += __data->data[sample][v] * _hP[core][h];
}

void RBMLayer::update_vBias0(int core, int sample){
	boost::mutex::scoped_lock  lock(_mutex_vBias0);
//	std::cout << "Core " << core << " update vBias0" << std::endl;
	for(int v=0; v<_vNeurons; v++)
		_vBias0[v] += __data->data[sample][v];
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
	for (int i = core; i < x_size; i += __cores)
		x[i] += __epsilon * (x0[i] - x1[i]) / __batch_size;
}

void RBMLayer::update2(int core, int x_size, int y_size, float **x, float **x0, float **x1) {
	for (int i = core; i < x_size; i += __cores)
		for (int j = 0; j < y_size; j++)
			x[i][j] += __epsilon * (x0[i][j] - x1[i][j]) / __batch_size;
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
//	std::cout << "synchonizing ..." << std::endl;
	boost::mutex::scoped_lock  lock(_mutex_ready);
	if (++__cores_ready == __cores){
		__cores_ready = 0;
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

	for (int v=core; v<_vNeurons; v+=__cores)
		for (int h=0; h<_hNeurons; h++)
			_W0[v][h] = _W1[v][h] = 0;

	for (int v=core; v<_vNeurons; v+=__cores)
		_vBias0[v] = _vBias1[v] = 0;

	for (int h=core; h<_hNeurons; h+=__cores)
		_hBias0[h] = _hBias1[h] = 0;

	synchronize();
}

void RBMLayer::operator()(int core) {
	std::cout << "Core " << core << " Starting ... " << std::endl;
	for (int sample = core; sample<__data->size; sample +=__cores){
//		std::cout << core << ": Sample " << sample << std::endl;

		up_d(core,sample);

		update_W0(core,sample);
		update_vBias0(core,sample);
		update_hBias0(core);

		down(core);
		up(core);

		update_W1(core);
		update_vBias1(core);
		update_hBias1(core);


		if ( ( (sample % __batch_size) - core) == (__batch_size - __cores) ||
				sample == (__data->size - (__data->size % __cores) + core - __cores) )
			update_weights(core);
	}
}
