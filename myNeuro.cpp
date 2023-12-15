#include "myNeuro.h"
//#include <QDebug>

#include <typeinfo>
#define STRING(Value) #Value

bool is_optimizedM;

//switchers
bool allow_optimisation_transform = false;
//bool allow_optimisation_transform = true;
//bool start_visualisation = false;
bool start_visualisation = true;

//bool allow_truncate_for_example = false;
bool allow_truncate_for_example = true;

double areaWidth = 1000.0;
double areaHeight = 1000.0;

int iCycle;
int iCycleTotal;

myNeuro::myNeuro()
{
   std::cout<<"\n_________________________________ start myNeuro cpp\n";;
    //--------многослойный
    inputNeurons = n1;
    outputNeurons =n3;

    nlCount = 2;//(!!!)укажи сколько слоев используешь, иначе нaвернется

    errLimit = errLimitG;

    errOptinizationLimit = errOptinizationLimitG;
    list = (nnLay*) malloc((nlCount)*sizeof(nnLay));



    inputs = (float*) malloc((inputNeurons)*sizeof(float));
    targets = (float*) malloc((outputNeurons)*sizeof(float));

    //list[0].setIO(100,20);
    //list[1].setIO(20,6);
    //list[2].setIO(6,3);
    //list[3].setIO(3,2);

    //list[0].setIO(n1, n2);
    //list[1].setIO(n2, 40);
    //list[1].setIO(40, n3);

    //list[0].setIO(n1,n2);
    //list[1].setIO(n2,60);
    //list[2].setIO(60,30);
    //list[3].setIO(30,n3);

//    list[0].setIO(n1, n2);
//    list[1].setIO(n2, 30);
//    list[2].setIO(30, n3);

    //list[0].setIO(n1, 1);
    //list[1].setIO(1, n3);

   /* list[0].setIO(80, 20);
    list[1].setIO(20, 10);*/

    list[0].setIO(n1, n2);
    list[1].setIO(n2, n3);

//   std::cout<<"\n_________________________________ start myNeuro cpp myNeuro\n";;


    //--------singel (пример)---------//
//    inputNeurons = 100;
//    outputNeurons =2;
//    nlCount = 2;
//    list = (nnLay*) malloc((nlCount)*sizeof(nnLay));
//    inputs = (float*) malloc((inputNeurons)*sizeof(float));
//    targets = (float*) malloc((outputNeurons)*sizeof(float));
//    list[0].setIO(100,10);
//    list[1].setIO(10,2);

}

float **  myNeuro::feedForwarding(bool mode_train)
{
//   std::cout<<"\n_________________________________ start myNeuro cpp feedForwarding\n";
    list[0].toHiddenLayer(inputs);
    for (int i =1; i<nlCount; i++)
        list[i].toHiddenLayer(list[i-1].getHidden());

    if (mode_train)//it is train mode
    {
        // printArray(list[3].getErrors(),list[3].getOutCount());
        return backPropagate();// обратн расп ошибки
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

        //return empy array^ just for compatability
        float ** err3f;
        err3f = (float**) malloc(sizeof(float)*nlCount);
        return err3f;
    }

}

void myNeuro::optimiseWay()
{
    //std::cout<<"\n_________________________________ optimiseWay!!!!!!!! \n";
    //start it now or later?
    if (!is_optimizedM)is_optimizedM = true;
}

float* myNeuro::processErrors(int i, bool & startOptimisation, bool showError = false, float totalE = 0.0)
{
    float  err1  = *list[i].getErrorsM();
    err1 = absF(err1);
    int lenLayer = list[i].getOutCount();

//    float sumErr = 0;
//    float p = err1;
//    for( int k = 0; k< lenLayer; k++ )
//        sumErr += p++;

//    float lenErr1 = err1[0];
//    float lenErr2 = err1[1];
//    float lenErr3 = err1[2];


    //startOptimisation = startOptimisation & (err1<errOptinizationLimit) ;
    startOptimisation = startOptimisation && (totalE<errOptinizationLimit) ; // & (i == (nlCount-1)) // & true ; //
    if (list[i].is_optimizedL != startOptimisation) list[i].is_optimizedL = startOptimisation;


    bool showBecuseOpt = (is_optimizedM || iCycle<3 );

    if (showError){
        std::cout << " layer:" + std::to_string(i);
        std::cout << " error:" + std::to_string(err1);
        //std::cout << " sum_delta_error:" + std::to_string(totalE);
        std::cout << " (optimisation:" + std::to_string((startOptimisation) ) + ")" ;
        std::cout <<  " (len:" + std::to_string(lenLayer ) + ")";
    }

    if(showError & !showBecuseOpt)std::cout <<  endl;

    if( (showError & showBecuseOpt) || is_optimizedM ){

        if(is_optimizedM)std::cout<<"\n_________________!is_optimizedM!________________\n";
//        std::cout<<"\n";
//        std::cout<<"err1 "<<lenErr1;
//        std::cout<<"err2 "<<lenErr2;
//        std::cout<<"err2 "<<lenErr3;

        std::cout<<"\n";
        //if(list[i].is_optimizedL )std::cout<<"\n_________________________________\n";
        std::cout<<" -layer:"+std::to_string(i)+" ("+std::to_string(lenLayer)+") ";
        printArray(list[i].getErrors(),i, lenLayer);
        //if(list[i].is_optimizedL )std::cout<<"\n_________________________________\n";
        std::cout<<"\n";
    }


// check size of list errors
//    float * e1 = list[i].getErrors();
//    cout<<','<<sizeof(e1)/sizeof(float) ;
//    return e1;

// empty response
//    float * e1;
//    return e1;

    return list[i].getErrors();

}

float ** myNeuro::backPropagate()
{   
    //   std::cout<<"\n_________________________________ start myNeuro cpp backPropagate\n";;
    //-------------------------------ERRORS-----CALC---------
    bool showError = false;
    bool startOptimisation = true;
    if(rand()%10000==9 || iCycle<3){
        showError = true;
    }

    //processErrors(nlCount,startOptimisation,showError);//how calc resul error?

    //start return sum of erros... why? just interesting
    float sum_out_error = list[nlCount-1].calcOutError(targets, showError);//for 1 out layer (where 2 output neurons for our case)



    //try to return array of errors up to start point, and calculate them
    //
//    float ** err3;
//    err3 = (float**) malloc((nlCount)*sizeof(float));//*2 malloc fail in counting mem
    float ** err3 = static_cast<float**> (malloc((nlCount)*sizeof(float)));
    err3[nlCount-1] = processErrors(nlCount-1,startOptimisation,showError,sum_out_error);
    //
    /////


    for (int i =nlCount-2; i>=0; i--){//for everyone over layer
        list[i].calcHidError(list[i+1].getErrors(),list[i+1].getMatrix(),
                             list[i+1].getInCount(),list[i+1].getOutCount(), showError);

        err3[i] = processErrors(i,startOptimisation,showError);

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
    return err3;
}

float ** myNeuro::train(float *in, float *targ, bool optimize)
{
//   std::cout<<"\n_________________________________ start myNeuro cpp train\n";;

    inputs = in;
    targets = targ;
    float ** resfeed = feedForwarding(true);
    if (optimize) {
        for (int i = (nlCount - 2); i >= 0; i--)
            sumFloatMD(i);
    }
    return resfeed;
}

void myNeuro::query(float *in)
{
   std::cout<<"\n_________________________________ start myNeuro cpp query\n";;
    inputs=in;
    feedForwarding(false);
    std::cout<<"\n_________________________________ end myNeuro cpp query\n";;
}

void myNeuro::sumFloatMD(int inS)
{
//    float* errors = static_cast<float*> (malloc((inS)*sizeof(float)*nlCount));
    float* errors = list[inS].getErrors();
//    float * errors = (float*) malloc((inS)*sizeof(float));

    for(int i=0; i < (list[inS].getOutCount()-1); i++)
    {
//        if(left[i] == NULL || right[i] == NULL)errors[i]=0;
//            errors[i] = left[i] + right[i];//absf?
//            errors[i] = absF(left[i]) + absF(right[i]);//absf?

        list[inS].errTmp[i] += errors[i];
    }
    //return errors;
};

void myNeuro::optimize_layer(int inS){
    int countOut = list[inS].getOutCount();
//    int countIn = list[i].getInCount();
if(rand()%100==9 )
    for(int inp =0; inp < countOut; inp++)
    {
      //if (rand() % 1000 == 9)
      //  std::cout << "truncMatrix layer:" << inS << " neuron:" << inp << " neuron val:" << list[inS].errTmp[inp] << "  countOut:" << countOut << "\n";;
        if(absF(list[inS].errTmp[inp])>1){
            std::cout<<"truncMatrix layer:"<< inS <<" neuron:"<< inp  <<" neuron val:"<< list[inS].errTmp[inp]  <<"  countOut:"<< countOut <<"\n";;
            list[inS].truncMatrixOut(inp);
            list[inS+1].truncMatrixIn(inp);
        }
    }
}

void myNeuro::write_matrix_var1(string file_name) {
    ofstream file(file_name.c_str(), ios::out);

    for (int i =nlCount-1; i>0; i--) {
        float ** w1 = list[i].getMatrix();
        for (int i = 1; i < list[i].getOutCount(); ++i) {
            for (int j = 1; j < list[i].getInCount(); ++j) {
                file << w1[i][j] << " ";
            }
            file << endl;
        }
    }

    file.close();
}

void myNeuro::printArray(float *arr, int iList, int s)
{
//    std::cout<<"printArray__\n";;
    std::string str_outErrors;
    for(int inp =0; inp < s; inp++)
    {
        std::string type_s;
        std::string str_f;
        type_s = typeid(arr[inp]).name();
        //std::cout<< type_s;
        str_f = 'f';
        if(type_s == str_f || type_s == "float") {

            //v0
            int i2 = 0;
            float N = absF(arr[inp]);
            float Nf = absF(arr[inp]);

//            std::cout << to_string(inp);
//            std::cout << "(_";

//            if(N < 0.9 and N > -0.9)

            while (N < 0.9 && i2<99)
            {
                N = N * 10; ++i2;
            }
            if (i2==99)i2=0;
            std::cout << i2 <<")";

            str_outErrors += to_string(inp) + '(' + to_string(Nf) + ')';

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

    cout << "\n" << fixed << str_outErrors;
}
