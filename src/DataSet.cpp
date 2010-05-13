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

void DataSet::read_idx(DataSet *dataset, const char* file_name) {
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

	dataset->size = msbchar_2_int(c_image);
	int rows = msbchar_2_int(c_row);
	int cols = msbchar_2_int(c_col);
	cout << "Rows: " << rows << "  Cols: " << cols << endl;
	dataset->dim = rows * cols;

	char *pixels = new char[dataset->size * dataset->dim];
	file.read(pixels, dataset->size * dataset->dim);
	file.close();

	dataset->data = new float*[dataset->size];
	for (int i = 0; i < dataset->size; i++) {
		dataset->data[i] = new float[dataset->dim];
		for (unsigned j = 0; j < dataset->dim; j++)
			dataset->data[i][j] = ((unsigned char) pixels[dataset->dim * i + j]) / 256.0;
	}
	delete pixels;
}

void DataSet::print(DataSet *dataset, int i, int cols) {
	for (int j=0; j<dataset->dim; j++){
		if ( j%cols == 0 ) cout << "\x1b[0m" << endl;
		cout << "\x1b[48;5;" << 255 - (int) (dataset->data[i][j] * 24) << "m ";
	}
	cout << "\x1b[0m" << endl;
}

void LabeledDataSet::read_idx_labels(LabeledDataSet *ldataset, const char *file_name) {
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
	if (n_labels != ldataset->size){
		cout << "The size of the labels don't match the size of the data." << endl;
		file.close();
		exit(EXIT_FAILURE);
	}

	ldataset->labels = new char[n_labels];
	file.read(ldataset->labels, n_labels);
	file.close();
}

void LabeledDataSet::read_idx(LabeledDataSet *ldataset, const char *data_file, const char *label_file){
	DataSet::read_idx(ldataset, data_file);
	LabeledDataSet::read_idx_labels(ldataset, label_file);
}
