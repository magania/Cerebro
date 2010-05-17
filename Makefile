INCLUDES=-Iinclude -I../boost_1_43_0
LIBS=-L ../boost_1_43_0/stage/lib/ -lboost_thread 

all:
	g++ $(INCLUDES)  -msse -mmmx  -c -o obj/DataSet.o src/DataSet.cpp
	g++ $(INCLUDES)  -msse -mmmx  -c -o obj/RBMLayer.o src/RBMLayer.cpp
	g++ $(INCLUDES) $(LIBS)  -msse -mmmx  -o test test.cpp obj/DataSet.o obj/RBMLayer.o
