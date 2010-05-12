INCLUDES=-Iinclude -I../boost_1_43_0
LIBS=

all:
	g++ $(INCLUDES) -c -o obj/DataSet.o src/DataSet.cpp
	g++ $(INCLUDES) -c -o obj/RBMLayer.o src/RBMLayer.cpp
	g++ $(INCLUDES) $(LIBS) -o test test.cpp obj/DataSet.o obj/RBMLayer.o -L ../boost_1_43_0/stage/lib/ -lboost_thread 