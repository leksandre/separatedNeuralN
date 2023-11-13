//#include <QCoreApplication>
//#include <QDebug>
//#include <QTime>
#include "myNeuro.h"

int main(int argc, char *argv[])
{
    //QCoreApplication a(argc, argv);

    myNeuro *bb = new myNeuro();
//myNeuro bb;
     //----------------------------------INPUTS----GENERATOR-------------
        //qsrand((QTime::currentTime().second()));
        float *abc = new float[100];
            for(int i=0; i<100;i++)
            {
            abc[i] =(rand()%98)*0.01+0.01;
            }

        float *cba = new float[100];
            for(int i=0; i<100;i++)
            {
            cba[i] =(rand()%98)*0.01+0.01;
            }

    //---------------------------------TARGETS----GENERATOR-------------
        float *tar1 = new float[2];
        tar1[0] =0.01;
        tar1[1] =0.99;
        float *tar2 = new float[2];
        tar2[0] =0.99;
        tar2[1] =0.01;

    //--------------------------------NN---------WORKING---------------
        bb->query(abc);
        std::cout<<"_________________________________";
        bb->query(cba);

        int i=0;
        while(i<100000)
        {
            bb->train(abc,tar1);
            bb->train(cba,tar2);
            i++;
        }

        std::cout<<"___________________RESULT_____________";
        bb->query(abc);
        std::cout<<"______";
        bb->query(cba);


        std::cout<<"_______________THE____END_______________";
       //std::cout<<"_______________THE____END_______________";

    	//return a.exec();
	return 0;
}
