/*
 * test.cpp
 *
 *  Created on: May 10, 2010
 *      Author: magania
 */
#include <DataSet.hpp>
#include <RBMLayer.hpp>

int main( int argc, const char* argv[] ){
	LabeledDataSet ldata("mnist/train-images-idx3-ubyte", "mnist/train-labels-idx1-ubyte");

//	for (int i=0; i< 10; i++)
//		ldata->print(i,28);

	RBMLayer layer1(ldata.dim, 8);

 	for (int t=1; t<2; t++){
 		std::cout << "Epoch: " << t << std::endl;
	     layer1.train(ldata, 1000, 0.1, 2);
 	}
}
