/*
 * test.cpp
 *
 *  Created on: May 10, 2010
 *      Author: magania
 */
#include <DataSet.hpp>
#include <RBMLayer.hpp>

#include <string>
#include <iostream>
#include <stdlib.h>

//	LabeledDataSet ldata("mnist/t10k-images-idx3-ubyte", "mnist/t10k-labels-idx1-ubyte");
int main( int argc, const char* argv[] ){
	LabeledDataSet ldata("mnist/train-images-idx3-ubyte", "mnist/train-labels-idx1-ubyte");

        std::cout << "layer1" << std::endl;
	RBMLayer layer1(ldata.dim, 1000);
	layer1.read_W("W_60.txt");

	std::cout << "Up data" << std::endl;
	DataSet* data2 = layer1.up_data(ldata);

        std::cout << "layer2" << std::endl;
	RBMLayer layer2(data2->dim, 1000);
        layer2.read_W("W2_60.txt");

	std::cout << "Up data" << std::endl;
	DataSet* data3 = layer2.up_data(*data2);

        std::cout << "layer3" << std::endl;
	RBMLayer layer3(data3->dim, 1000);
	
        for (int t=1; t<=60; t++){
            std::cout << "Epoch: " << t << std::endl;
	    layer3.train(*data3, 1000, 0.1, 8);
	    if ( t % 10 == 1 || t == 60 ){
	    	std::stringstream file;
	    	file << "W3_" << t << ".txt";
	    	layer1.write_W(file.str().c_str());
	    }
 	}
}
