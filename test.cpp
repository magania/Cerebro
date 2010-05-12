/*
 * test.cpp
 *
 *  Created on: May 10, 2010
 *      Author: magania
 */
#include <DataSet.hpp>
#include <RBMLayer.hpp>

int main( int argc, const char* argv[] ){
	LabeledDataSet data("mnist/train-images-idx3-ubyte", "mnist/train-labels-idx1-ubyte");

//	for (int i=0; i< data.size(); i++)
//		data.print(i,28);

	RBMLayer layer1(&data,1);

	layer1.train(10, 0.1, 2);

}
