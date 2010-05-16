INCLUDES=-Iinclude -I../boost_1_43_0
LIBS=-L ../boost_1_43_0/stage/lib/ -lboost_thread 

all:
	g++ $(INCLUDES) -m32 -c -o obj/DataSet.o src/DataSet.cpp
	g++ $(INCLUDES) -msse -mmmx -m32 -c -o obj/RBMLayer.o src/RBMLayer.cpp
	g++ $(INCLUDES) $(LIBS) -msse -mmmx -m32 -o test test.cpp obj/DataSet.o obj/RBMLayer.o
