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

DataSet::DataSet(int data_size, int dimension){
	size = data_size;
	dim = dimension;

	data = (float**) malloc(size * sizeof(float*));
	for (int i = 0; i < size; i++)
		posix_memalign((void **) &data[i], 16, dim * sizeof(float));
}

DataSet::~DataSet(){
	for (int i = 0; i < size; i++)
		free(data[i]);
	free(data);
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

    size = msbchar_2_int(c_image);
	int rows = msbchar_2_int(c_row);
	int cols = msbchar_2_int(c_col);
	dim = rows*cols;
	cout << "Rows: " << rows << "  Cols: " << cols << endl;

	data = (float**) malloc(size * sizeof(float*));
	for (int i = 0; i < size; i++)
		posix_memalign((void **) &data[i], 16, dim * sizeof(float));

	char *pixels = new char[size * dim];
	file.read(pixels, size * dim);
	file.close();

	for (int i = 0; i < size; i++)
		for (int j = 0; j < dim; j++)
			data[i][j] = ((unsigned char) pixels[dim * i + j]) / 256.0;

	delete [] pixels;
}

void DataSet::print(int i, int cols) {
	for (int j=0; j<dim; j++){
		if ( j%cols == 0 ) cout << "\x1b[0m" << endl;
		cout << "\x1b[48;5;" << 255 - (int) (data[i][j] * 24) << "m ";
	}
	cout << "\x1b[0m" << endl;
}


LabeledDataSet::LabeledDataSet(const char *data_file, const char *label_file) :
		DataSet(data_file)
	{
	ifstream file(label_file, ios::in | ios::binary);
	if (!file.is_open()) {
		cout << "Unable to open file. " << label_file << endl;
		exit(EXIT_FAILURE);
	}
	char c_magic[4], c_label[4];
	int magic;

	file.read(c_magic, 4);
	if (msbchar_2_int(c_magic) != 2049) {
		cout << "Invalid magic number. " << label_file << endl;
		file.close();
		exit(EXIT_FAILURE);
	}

	file.read(c_label, 4);

	int n_labels = msbchar_2_int(c_label);
	if (n_labels != size){
		cout << "The size of the labels don't match the size of the data." << endl;
		file.close();
		exit(EXIT_FAILURE);
	}

	labels = new char[n_labels];
	file.read(labels, n_labels);
	file.close();
}

LabeledDataSet::~LabeledDataSet(){
	delete [] labels;
}
