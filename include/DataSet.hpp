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
	DataSet(const char *file_name);
	DataSet(int size, int dim);
	virtual ~DataSet();
	int size, dim;
	float **data;

	void print(int i, int cols = 1);

	static int msbchar_2_int(char* msbchar);
};

class LabeledDataSet : public DataSet {
public:
	LabeledDataSet(const char *data_files, const char *label_file);
	LabeledDataSet(int size, int dim);
	~LabeledDataSet();
	char *labels;
};


#endif /* DATASET_HPP_ */
