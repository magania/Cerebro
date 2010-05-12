/*
 * DataSet.hpp
 *
 *  Created on: May 7, 2010
 *      Author: magania
 */

#ifndef DATASET_HPP_
#define DATASET_HPP_


class DataSet {
private:
	int _size, _dim;
	float **_data;

protected:
	int msbchar_2_int(char* msbchar);

public:
	DataSet(const char *file_name);

	float *get(int i);
	int size();
	int dim();

	void print(int i, int cols);
};

class LabeledDataSet : public DataSet {
private:
	char *_labels;
	void read_idx_labels(const char *file_name);

public:
	LabeledDataSet(const char *data_files, const char *label_file);
	int* label();
};


#endif /* DATASET_HPP_ */
