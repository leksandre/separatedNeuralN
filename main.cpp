//#include <QCoreApplication>
//#include <QDebug>
//#include <QTime>


//for linux
//#include "myNeuro.cpp"
//#include <sys/time.h>

//for win!!
#include "myNeuro.h"
#include <time.h>

int iCycle;
int main(int argc, char *argv[])
{
    //QCoreApplication a(argc, argv);
//   std::cout<<"\n_________________________________ start main 0\n";;;
    time_t start, end;
    time(&start);

    myNeuro *bb = new myNeuro();
//    return 0;
//myNeuro bb;
     //----------------------------------INPUTS----GENERATOR-------------
//   std::cout<<"\n_________________________________ start main\n";;;
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
        std::cout<<"\n________________TARGETS----GENERATOR_________________\n";;
        float *tar1 = new float[2];
        tar1[0] =0.01;
        tar1[1] =0.99;
        float *tar2 = new float[2];
        tar2[0] =0.99;
        tar2[1] =0.01;

    //--------------------------------NN---------WORKING---------------

        std::cout<<"\n___________________calculate_without_train_____________\n";;
        bb->query(abc);
        bb->query(cba);

        std::cout<<"\n________________start_train_________________\n";;
        iCycle = 0;
        while(iCycle<100000)
        {
            bb->train(abc,tar1);
            bb->train(cba,tar2);
            iCycle++;
        }

        std::cout<<"\n___________________calculate_RESULT_____________\n";;
        bb->query(abc);
        std::cout<<"______\n";;
        bb->query(cba);


        std::cout<<"\n_______________THE____END_______________\n";;
       //std::cout<<"\n_______________THE____END_______________\n";;

    	//return a.exec();

    time(&end);

    // Calculating total time taken by the program.
    double time_taken;
    time_taken = double(end - start);
    std::cout << "Time taken by program is : " << time_taken << "";
    std::cout << " sec " << "\n";

    return 0;
}
