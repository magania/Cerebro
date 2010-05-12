/*
 * DataSet.cpp
 *
 *  Created on: May 10, 2010
 *      Author: magania
 */

#include <iostream>
#include <fstream>
#include <cstdlib>

#include <DataSet.hpp>

using namespace std;

int DataSet::msbchar_2_int(char* msbchar) {
	char lsbchar[4];
	lsbchar[3] = msbchar[0];
	lsbchar[2] = msbchar[1];
	lsbchar[1] = msbchar[2];
	lsbchar[0] = msbchar[3];
	return *((int*) lsbchar);
}

DataSet::DataSet(const char* file_name) {
	ifstream file(file_name, ios::in | ios::binary);
	if (!file.is_open()) {
		cout << "Unable to open file. " << file_name << endl;
		exit(EXIT_FAILURE);
	}

	char c_magic[4], c_image[4], c_row[4], c_col[4];
	int magic;

	file.read(c_magic, 4);
	if (msbchar_2_int(c_magic) != 2051) {
		cout << "Invalid file format. " << file_name << endl;
		file.close();
		exit(EXIT_FAILURE);
	}

	file.read(c_image, 4);
	file.read(c_row, 4);
	file.read(c_col, 4);

	_size = msbchar_2_int(c_image);
	int rows = msbchar_2_int(c_row);
	int cols = msbchar_2_int(c_col);
	cout << "Rows: " << rows << "  Cols: " << cols << endl;
	_dim = rows * cols;

	char *pixels = new char[_size * _dim];
	file.read(pixels, _size * _dim);
	file.close();

	_data = new float*[_size];
	for (int i = 0; i < _size; i++) {
		_data[i] = new float[_dim];
		for (unsigned j = 0; j < _dim; j++)
			_data[i][j] = ((unsigned char) pixels[_dim * i + j]) / 256.0;
	}
	delete pixels;
}

float *DataSet::get(int i) {
	return _data[i];
}

int DataSet::size() {
	return _size;
}

int DataSet::dim(){
	return _dim;
}

void DataSet::print(int i, int cols) {
	for (int j=0; j<_dim; j++){
		if ( j%cols == 0 ) cout << "\x1b[0m" << endl;
		cout << "\x1b[48;5;" << 255 - (int) (_data[i][j] * 24) << "m ";
	}
	cout << "\x1b[0m" << endl;
}

void LabeledDataSet::read_idx_labels(const char *file_name) {
	ifstream file(file_name, ios::in | ios::binary);
	if (!file.is_open()) {
		cout << "Unable to open file. " << file_name << endl;
		exit(EXIT_FAILURE);
	}
	char c_magic[4], c_label[4];
	int magic;

	file.read(c_magic, 4);
	if (msbchar_2_int(c_magic) != 2049) {
		cout << "Invalid magic number. " << file_name << endl;
		file.close();
		exit(EXIT_FAILURE);
	}

	file.read(c_label, 4);

	int n_labels = msbchar_2_int(c_label);

	_labels = new char[n_labels];
	file.read(_labels, n_labels);
	file.close();
}

LabeledDataSet::LabeledDataSet(const char *data_files, const char *label_file) :
	DataSet(data_files) {
	read_idx_labels(label_file);
}
