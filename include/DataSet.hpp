/*
 * DataSet.hpp
 *
 *  Created on: May 7, 2010
 *      Author: magania
 */

#ifndef DATASET_HPP_
#define DATASET_HPP_

class DataSet {
public:
	int size, dim;
	float **data;

	static void read_idx(DataSet *dataset, const char *file_name);
	static void print(DataSet *dataset, int i, int cols = 1);

	static int msbchar_2_int(char* msbchar);
};

class LabeledDataSet : public DataSet {
public:
	char *labels;

	static void read_idx(LabeledDataSet *ldataset, const char *data_files, const char *label_file);
private:
	static void read_idx_labels(LabeledDataSet *ldataset, const char *file_name);
};


#endif /* DATASET_HPP_ */
