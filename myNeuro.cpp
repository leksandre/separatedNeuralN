#include "myNeuro.h"
//#include <QDebug>
//using namespace std;

#define STRING(Value) #Value

myNeuro::myNeuro()
{
   std::cout<<"\n_________________________________ start myNeuro cpp\n";;
    //--------многослойный
    inputNeurons = 100;
    outputNeurons =2;
    nlCount = 4;
    errLimit = 0.000005;
    errOptinizationLimit = 0.00003;
    list = (nnLay*) malloc((nlCount)*sizeof(nnLay));

    inputs = (float*) malloc((inputNeurons)*sizeof(float));
    targets = (float*) malloc((outputNeurons)*sizeof(float));

    list[0].setIO(100,20);
//    return ;
    list[1].setIO(20,6);
    list[2].setIO(6,3);
    list[3].setIO(3,2);

//   std::cout<<"\n_________________________________ start myNeuro cpp myNeuro\n";;


    //--------однослойный---------//
//    inputNeurons = 100;
//    outputNeurons =2;
//    nlCount = 2;
//    list = (nnLay*) malloc((nlCount)*sizeof(nnLay));

//    inputs = (float*) malloc((inputNeurons)*sizeof(float));
//    targets = (float*) malloc((outputNeurons)*sizeof(float));

//    list[0].setIO(100,10);
//    list[1].setIO(10,2);

}

void myNeuro::feedForwarding(bool ok)
{
//   std::cout<<"\n_________________________________ start myNeuro cpp feedForwarding\n";
    list[0].makeHidden(inputs);
    for (int i =1; i<nlCount; i++)
        list[i].makeHidden(list[i-1].getHidden());

    if (!ok) // is query mode
    {
//        std::cout<<std::to_string(outputNeurons)+"!ok - Feed Forward: \n";;
//        std::cout<<"nlCount:"+std::to_string(nlCount)+"\n";
//        std::cout<<"total outputNeurons:"+std::to_string(outputNeurons)+"\n";

        for(int out =0; out < outputNeurons; out++)
        {
            std::cout<<"outputNeuron "+std::to_string(out)+":";
            float outit = list[nlCount-1].hidden[out];
          std::cout<<std::to_string(outit)+"\n";
        }
        return;
    }
    else //it is train mode
    {
       // printArray(list[3].getErrors(),list[3].getOutCount());
        backPropagate();
    }
}

void myNeuro::optimiseWay()
{
//    std::cout<<"\n_________________________________ optimiseWay!!!!!!!! \n";
}

void myNeuro::processErrors(int i, bool & startOptimisation, bool showError)
{
    float  err1 = *list[i].getErrorsM();
    startOptimisation = startOptimisation & (err1<errOptinizationLimit);
    if (showError)
        std::cout << ", i:" + std::to_string(i) + " " + std::to_string( err1 ) ;
}

    void myNeuro::backPropagate()
{   
    //   std::cout<<"\n_________________________________ start myNeuro cpp backPropagate\n";;
    //-------------------------------ERRORS-----CALC---------
    bool showError = false;
    bool startOptimisation = true;
    if(rand()%10000==9){
        showError = true;
    }

    list[nlCount-1].calcOutError(targets);

    processErrors(nlCount-1,startOptimisation,showError);

    for (int i =nlCount-2; i>=0; i--){
        list[i].calcHidError(list[i+1].getErrors(),list[i+1].getMatrix(),
                             list[i+1].getInCount(),list[i+1].getOutCount());

        processErrors(i,startOptimisation,showError);

    }

    if(showError)std::cout<<"\n";

    if(startOptimisation){
        optimiseWay();
    }

    //-------------------------------UPD-----WEIGHT---------
    for (int i =nlCount-1; i>0; i--)
        list[i].updMatrix(list[i-1].getHidden());
    list[0].updMatrix(inputs);
}

void myNeuro::train(float *in, float *targ)
{
//   std::cout<<"\n_________________________________ start myNeuro cpp train\n";;
    inputs = in;
    targets = targ;
    feedForwarding(true);
}

void myNeuro::query(float *in)
{
   std::cout<<"\n_________________________________ start myNeuro cpp query\n";;
    inputs=in;
    feedForwarding(false);
}

void myNeuro::printArray(float *arr, int s)
{
    std::cout<<"printArray__\n";;
    for(int inp =0; inp < s; inp++)
    {
        std::cout<<arr[inp];
    }
}
