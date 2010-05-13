/*
 * test.cpp
 *
 *  Created on: May 10, 2010
 *      Author: magania
 */
#include <DataSet.hpp>
#include <RBMLayer.hpp>

int main( int argc, const char* argv[] ){
	LabeledDataSet *ldata = new LabeledDataSet();
	LabeledDataSet::read_idx(ldata, "mnist/train-images-idx3-ubyte", "mnist/train-labels-idx1-ubyte");

//	for (int i=0; i< data.size(); i++)
//		data.print(i,28);

	RBMLayer layer1(ldata->dim,3);

	layer1.train(ldata, 100, 0.1, 2);

}
