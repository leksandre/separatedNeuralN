#include "myNeuro.h"
//#include <QDebug>
//using namespace std;
#include <typeinfo>
#define STRING(Value) #Value

myNeuro::myNeuro()
{
   std::cout<<"\n_________________________________ start myNeuro cpp\n";;
    //--------многослойный
    inputNeurons = 100;
    outputNeurons =2;
    nlCount = 4;
    errLimit = 0.000005;
    couldoptimizeM = false;
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

void myNeuro::feedForwarding(bool mode_train)
{
//   std::cout<<"\n_________________________________ start myNeuro cpp feedForwarding\n";
    list[0].toHiddenLayer(inputs);
    for (int i =1; i<nlCount; i++)
        list[i].toHiddenLayer(list[i-1].getHidden());

    if (mode_train)//it is train mode
    {
        // printArray(list[3].getErrors(),list[3].getOutCount());
        backPropagate();// обратн расп ошибки
    } else // is query mode
    {
//        std::cout<<std::to_string(outputNeurons)+"!mode_train - Feed Forward: \n";;
//        std::cout<<"nlCount:"+std::to_string(nlCount)+"\n";
//        std::cout<<"total outputNeurons:"+std::to_string(outputNeurons)+"\n";

        for (int out = 0; out < outputNeurons; out++) {
            std::cout << "outputNeuron " + std::to_string(out) + ":";
            float outit = list[nlCount - 1].hidden[out];
            std::cout << std::to_string(outit) + "\n";
        }
        return;
    }

}

void myNeuro::optimiseWay()
{
//    std::cout<<"\n_________________________________ optimiseWay!!!!!!!! \n";
    //start it now or later?
    couldoptimizeM = true;
}

void myNeuro::processErrors(int i, bool & startOptimisation, bool showError)
{
    float  err1 = *list[i].getErrorsM();
    startOptimisation = startOptimisation & (err1<errOptinizationLimit) & true ; // & (i == 3)

    if (list[i].couldoptimizeL != startOptimisation) list[i].couldoptimizeL = startOptimisation;

    if (showError)
        std::cout << " layer:" + std::to_string(i) + " error:" + std::to_string(err1) + " (" +
//                "" + std::to_string((startOptimisation) ) + ") (" +
                     std::to_string(list[i].getInCount()) + ") ";
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

    list[nlCount-1].calcOutError(targets);//for 1 out layer (where 2 output neurons for our case)

    processErrors(nlCount-1,startOptimisation,showError);

    for (int i =nlCount-2; i>=0; i--){//for everyone over layer
        list[i].calcHidError(list[i+1].getErrors(),list[i+1].getMatrix(),
                             list[i+1].getInCount(),list[i+1].getOutCount());

        processErrors(i,startOptimisation,showError);
        if(showError & couldoptimizeM){
            std::cout<<"\n";
            if(list[i].couldoptimizeL )std::cout<<"\n_________________________________\n";
            std::cout<<" layer:"+std::to_string(i)+" ";
            printArray(list[i+1].getErrors(),i, 100);
            if(list[i].couldoptimizeL )std::cout<<"\n_________________________________\n";
            std::cout<<"\n";
        }

    }

    if(showError) {
        std::cout<<"\n";
    }

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

void myNeuro::printArray(float *arr, int iList, int s)
{
//    std::cout<<"printArray__\n";;
    for(int inp =0; inp < s; inp++)
    {
        std::string type_s;
        std::string str_f;
        type_s = typeid(arr[inp]).name();
        //std::cout<< type_s;
        str_f = 'f';
        if(type_s == str_f | type_s == "float") {

            int i2 = 0;
            float N = arr[inp];
            while (N > 0)
            {
                N = N / 10; ++i2;
            }

            //v0
            std::cout<< i2;

            //v1
            //std::cout<< round(arr[inp]*1000000);


            //v2
            //float errSigm = list[iList].sigmoida(arr[inp]);
            //std::string errSigmS = std::to_string(errSigm);
            //if (errSigm == 0.5 | errSigmS == "0.5")errSigm = 0;
            //std::cout << errSigm;
     
        }else{
            std::cout<< (arr[inp]);
        }

        std::cout<< ',';
    }
}
