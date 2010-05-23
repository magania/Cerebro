/*
 * RBMLayer.cpp
 *
 *  Created on: May 7, 2010
 *      Author: magania
 */

#include <RBMLayer.hpp>

RBMLayer::RBMLayer(int visible_neurons, int hidden_neurons) :
	_aleatorio(rng, uniform_number), _vNeurons(visible_neurons), _hNeurons(
			hidden_neurons) {
//	std::cout << "RBMLayer constructor ... " << std::endl;
	_W = (float**) malloc(_vNeurons * sizeof(float*));
	_W0 = (float**) malloc(_vNeurons * sizeof(float*));
	_W1 = (float**) malloc(_vNeurons * sizeof(float*));
	for (int i=0; i < _vNeurons; i++) {
		posix_memalign((void **) &_W[i], 16, _hNeurons * sizeof(float));
		posix_memalign((void **) &_W0[i], 16, _hNeurons * sizeof(float));
		posix_memalign((void **) &_W1[i], 16, _hNeurons * sizeof(float));
		//zero(_hNeurons, _W[i]);
		for (int j=0; j < _hNeurons; j++)
			_W[i][j] = _aleatorio() * 0.1;
		zero(_hNeurons, _W0[i]);
		zero(_hNeurons, _W1[i]);
	}

	posix_memalign((void **) &_vBias, 16, _vNeurons * sizeof(float));
	posix_memalign((void **) &_vBias0, 16, _vNeurons * sizeof(float));
	posix_memalign((void **) &_vBias1, 16, _vNeurons * sizeof(float));

	posix_memalign((void **) &_hBias, 16, _hNeurons * sizeof(float));
	posix_memalign((void **) &_hBias0, 16, _hNeurons * sizeof(float));
	posix_memalign((void **) &_hBias1, 16, _hNeurons * sizeof(float));

	zero(_vNeurons, _vBias);
	zero(_vNeurons, _vBias0);
	zero(_vNeurons, _vBias1);

	zero(_hNeurons, _hBias);
	zero(_hNeurons, _hBias0);
	zero(_hNeurons, _hBias1);
}

RBMLayer::~RBMLayer() {
//	std::cout << "RBMLayer destructor ..." << std::endl;

	for (int i=0; i < _vNeurons; i++) {
		free(_W[i]);
		free(_W0[i]);
		free(_W1[i]);
	}
	free(_W);
	free(_W0);
	free(_W1);

	free(_vBias);
	free(_vBias0);
	free(_vBias1);

	free(_hBias);
	free(_hBias0);
	free(_hBias1);
}

/* x = x + k*a; */
void RBMLayer::mul(int size, float* x, float* a, float k) {
	int nLoop = size / 4;

	__m128 K = _mm_set_ps1(k);
	__m128 AB;

	__m128* A = (__m128*) a;
	__m128* X = (__m128*) x;

	for(int i=0; i<nLoop; i++){
		AB = _mm_mul_ps(*A++,K);
		*X++ = _mm_add_ps(*X, AB);
	}
}

	/* a = a+b */
void RBMLayer::add(int size, float* a, float* b) {
	int nLoop = size / 4;

	__m128* A = (__m128*) a;
	__m128* B = (__m128*) b;

	for(int i=0; i<nLoop; i++)
		*A++ = _mm_add_ps(*A, *B++);
}

	/* x = x + k*(a-b) */
void RBMLayer::del(int size, float* x, float* a, float* b, float k) {
	int nLoop = size / 4;

	__m128 K = _mm_set_ps1(k);
	__m128 AB;

	__m128* A = (__m128*) a;
	__m128* B = (__m128*) b;
	__m128* X = (__m128*) x;

	for(int i=0; i<nLoop; i++) {
		AB = _mm_sub_ps(*A++, *B++);
		AB = _mm_mul_ps(AB, K);
		*X++ = _mm_add_ps(*X, AB);
	}
}

	/* x = 0 */
void RBMLayer::zero(int size, float *x) {
	int nLoop = size / 4;

	__m128 zero = _mm_set_ps1(0);
	__m128* X = (__m128*) x;

	for(int i=0; i<nLoop; i++)
		*X++ = _mm_mul_ps(*X, zero);
}

void RBMLayer::train(DataSet& data, int batch_size, float epsilon, int cores) {
//	std::cout << "RBMLayer train: cores " << cores << " epsilon " << epsilon
//			<< std::endl;
	__data = &data;
	__batch_size = batch_size;
	__epsilon = epsilon;
	__cores_ready = 0;
	__cores = cores;

	_hP = (float**) malloc(__cores * sizeof(float*));
	_vP = (float**) malloc(__cores * sizeof(float*));
	for (int i = 0; i < __cores; i++) {
		posix_memalign((void **) &_vP[i], 16, _vNeurons * sizeof(float));
		posix_memalign((void **) &_hP[i], 16, _hNeurons * sizeof(float));
		zero(_hNeurons, _hP[i]);
		zero(_vNeurons, _vP[i]);
	}

	boost::thread* threads[__cores];
	for (int i = 0; i < __cores; i++)
		threads[i] = new boost::thread(boost::ref(*this), i);

	for (int i = 0; i < __cores; i++)
		threads[i]->join();

	for (int i = 0; i < __cores; i++)
		delete threads[i];

/*	for (int h = 0; h < _hNeurons; h++)
		std::cout << "Bias " << _hBias[h] << std::endl;

	DataSet xdata(_hNeurons, _vNeurons);

	float min = _W[0][0];
	float max = _W[0][0];
	for (int v = 0; v < _vNeurons; v++)
		for (int h = 0; h < _hNeurons; h++) {
			if (_W[v][h] < min)
				min = _W[v][h];
			if (_W[v][h] > max)
				max = _W[v][h];
		}
	std::cout << "Min: " << min << " Max: " << max << std::endl;

	for (int v=0; v<_vNeurons; v++)
	  for (int h=0; h<_hNeurons; h++)
	    xdata.data[h][v] = (_W[v][h] - min)/(max-min);

	 xdata.print(0, 28);
	 xdata.print(1, 28);*/

	for (int i = 0; i < __cores; i++) {
		free(_vP[i]);
		free(_hP[i]);
	}
	free(_vP);
	free(_hP);
}

void RBMLayer::write_W(const char* file_name){
	std::cout << "Writing " << file_name << std::endl;
	std::ofstream file(file_name);
	for (int h = 0; h < _hNeurons; h++){
		for (int v = 0; v < _vNeurons; v++)
			file << _W[v][h] << ' ';
		file << std::endl;
	}

	for (int h = 0; h < _hNeurons; h++)
		file << _hBias[h] << ' ';
	file << std::endl;

	for (int v = 0; v < _vNeurons; v++)
			file << _vBias[v] << ' ';
	file << std::endl;

}

void RBMLayer::read_W(const char* file_name){
	std::cout << "Reading " << file_name << std::endl;
	std::ifstream file(file_name);
	for (int h = 0; h < _hNeurons; h++)
		for (int v = 0; v < _vNeurons; v++)
			file >> _W[v][h];

	for (int h = 0; h < _hNeurons; h++)
		file >> _hBias[h];

	for (int v = 0; v < _vNeurons; v++)
			file >> _vBias[v];

}

inline void RBMLayer::up(int core, float *V) {
	for (int h = 0; h < _hNeurons; h++) {
		float sum = 0;
		for (int v = 0; v < _vNeurons; v++)
			sum += _W[v][h] * V[v];
		_hP[core][h] = 1.0 / (1.0 + exp(-sum - _hBias[h]));
	}
}

void RBMLayer::up_d(int core, int sample) {
	//	std::cout << "Core " << core << " up_d" << std::endl;
	up(core, __data->data[sample]);
}

DataSet* RBMLayer::up_data(DataSet& data) {
	_hP = (float**) malloc(1 * sizeof(float*));
	DataSet* out_data = new DataSet(data.size, _hNeurons);

	for (int sample = 0; sample < data.size; sample++){
		_hP[0] = out_data->data[sample];
		up(0, data.data[sample]);
	}

	free(_hP);
	return out_data;
}

DataSet* RBMLayer::down_data(DataSet& data) {
	_vP = (float**) malloc(1 * sizeof(float*));
	_hP = (float**) malloc(1 * sizeof(float*));
	DataSet* out_data = new DataSet(data.size, _vNeurons);
	for (int sample = 0; sample < 100; sample++){
		_vP[0] = out_data->data[sample];
		_hP[0] = data.data[sample];
		down(0);
	}
	free(_vP);
	free(_hP);
	return out_data;
}

void RBMLayer::up(int core) {
	//	std::cout << "Core " << core << " up" << std::endl;
	up(core, _vP[core]);
}

inline float RBMLayer::sample(float x) {
	return x > _aleatorio();
}

void RBMLayer::down(int core) {
	// std::cout << "Core " << core << " down" << std::endl;
	for (int v = 0; v < _vNeurons; v++) {
		float sum = 0;
		for (int h = 0; h < _hNeurons; h++)
			sum += _W[v][h] * sample(_hP[core][h]);
		_vP[core][v] = 1.0 / (1.0 + exp(-sum - _vBias[v]));
	}
}

void RBMLayer::update_W0(int core, int sample) {
	// std::cout << "Core " << core << " update W0" << std::endl;
	boost::mutex::scoped_lock lock(_mutex_W0);
	for (int v = 0; v < _vNeurons; v++)
		mul(_hNeurons, _W0[v], _hP[core], __data->data[sample][v]);
}

void RBMLayer::update_vBias0(int core, int sample) {
	// std::cout << "Core " << core << " update vBias0" << std::endl;
	boost::mutex::scoped_lock lock(_mutex_vBias0);
	add(_vNeurons, _vBias0, __data->data[sample]);
}

void RBMLayer::update_hBias0(int core) {
	//	std::cout << "Core " << core << " update hBias0" << std::endl;
	boost::mutex::scoped_lock lock(_mutex_hBias0);
	add(_hNeurons, _hBias0, _hP[core]);
}

void RBMLayer::update_W1(int core) {
	//	std::cout << "Core " << core << " update W1" << std::endl;
	boost::mutex::scoped_lock lock(_mutex_W1);
	for (int v = 0; v < _vNeurons; v++)
		mul(_hNeurons, _W1[v], _hP[core], _vP[core][v]);
}

void RBMLayer::update_vBias1(int core) {
	//	std::cout << "Core " << core << " update vBias1" << std::endl;
	boost::mutex::scoped_lock lock(_mutex_vBias1);
	add(_vNeurons, _vBias1, _vP[core]);
}

void RBMLayer::update_hBias1(int core) {
	//	std::cout << "Core " << core << " update hBias1" << std::endl;
	boost::mutex::scoped_lock lock(_mutex_hBias1);
	add(_hNeurons, _hBias1, _hP[core]);
}

void RBMLayer::update1(int core, int x_size, float *x, float *x0, float *x1) {
	float ab = __epsilon/ __batch_size;
	for (int i = core; i < x_size; i += __cores)
		x[i] += ab * (x0[i] - x1[i]);
}

void RBMLayer::update2(int core, int x_size, int y_size, float **x, float **x0, float **x1) {
	for (int i = core; i < x_size; i += __cores)
		del(y_size, x[i], x0[i], x1[i], __epsilon / __batch_size);
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
	boost::mutex::scoped_lock lock(_mutex_ready);
	if (++__cores_ready == __cores) {
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

	synchronize();

	if (core == 0) {
	for (int i=0; i < _vNeurons; i++) {
		zero(_hNeurons, _W0[i]);
		zero(_hNeurons, _W1[i]);
	}

	zero(_vNeurons, _vBias0);
	zero(_vNeurons, _vBias1);

	zero(_hNeurons, _hBias0);
	zero(_hNeurons, _hBias1);
	}

	synchronize();
}

void RBMLayer::operator()(int core) {
//	std::cout << "Core " << core << " Starting ... " << std::endl;
	for (int sample = core; sample < __data->size; sample += __cores) {
//		std::cout << core << ": Sample " << sample << std::endl;

		up_d(core, sample);

		update_W0(core, sample);
		update_vBias0(core, sample);
		update_hBias0(core);

		down(core);
		up(core);

		update_W1(core);
		update_vBias1(core);
		update_hBias1(core);

		if (((sample % __batch_size) - core) == (__batch_size - __cores))
				//|| sample == (__data->size - (__data->size % __cores) + core - __cores))
			update_weights(core);
	}
}
