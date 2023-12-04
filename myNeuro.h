#ifndef MYNEURO_H
#define MYNEURO_H
#include <iostream>
#include <math.h>
//#include <sdk_dev/math.h>
//#include <QtGlobal>
//#include <QDebug>



#include <fstream>
#include <cstring>
#include <string>
#include <cstdio>
#include <cstdlib>
//#include <cmath>
#include <vector>
#include <set>
#include <iterator>
#include <algorithm>


using namespace std;
// Training image file name
const string training_image_fn = "mnist/train-images.idx3-ubyte";

// Training label file name
const string training_label_fn = "mnist/train-labels.idx1-ubyte";

// Weights file name
const string model_fn = "model-neural-network.dat";

// Report file name
const string report_fn = "training-report.dat";

// Number of training samples
const int nTraining = 60000; // 10000;//

// Image size in MNIST database
const int width = 28;
const int height = 28;
const int n1 = width * height; // = 784, without bias neuron 
const int n2 = 128;
const int n3 = 10; // Ten classes: 0 - 9

const float errLimitG = 0.000005;

const float errOptinizationLimitG = 0.0001; //0.00003; //0.000001; //0.00003;



//for win!!
//
#include <sstream>
#include <string>
template<class T>
std::string toString(const T& value) {
    std::ostringstream os;
    os << value;
    return os.str();
}
//
//for win



 couldoptimizeM;
 iCycle;
 iCycleTotal;
double absD(double N);
float absF(float N);


#define learnRate 0.1
#define randWeight (( ((float)rand() / (float)RAND_MAX) - 0.5)* pow(out,-0.5))
class myNeuro
{
public:
    myNeuro();



    struct nnLay{
           int in;
           int out;

           bool couldoptimizeL;

           float** matrix;
           float* hidden;
           float* errors;
           int getInCount(){return in;}
           int getOutCount(){return out;}
           float **getMatrix(){return matrix;}
           float *getErrorsM(){return errors;}
           void updMatrix(float *enteredVal)
           {
               for(int ou =0; ou < out; ou++)
               {

                   for(int hid =0; hid < in; hid++)
                   {
                       matrix[hid][ou] += (learnRate * errors[ou] * enteredVal[hid]);
                   }
                   matrix[in][ou] += (learnRate * errors[ou]);
               }
           };
           void setIO(int inputs, int outputs)
           {
               in=inputs;
               out=outputs;

               std::cout << " in-out " + std::to_string(in) + " - " + std::to_string(out) + " \n ";
               std::cout << " randWeight " + std::to_string(randWeight) + " \n ";


               hidden = (float*) malloc((out)*sizeof(float));

               matrix = (float**) malloc((in+1)*sizeof(float)*2);//*2 malloc fail in counting mem
               for(int inp =0; inp < in+1; inp++)
               {
                   try {
                   matrix[inp] = (float*) malloc(out*sizeof(float));
                   }
                   catch (const std::out_of_range& e) {
                       std::cout << "Out of Range error.1\n";;
                       std::cerr << e.what();
                   } catch (const std::exception& e) {
                           std::cout << "Out of Range error.01\n";;
                           std::cerr << e.what();
                   } catch (const std::string& e) {
                           std::cout << "Out of Range error.10\n";;
                           //std::cerr << e.what();
                   } catch (...) {
                           std::cout << "Out of Range error.11\n";;
                   }
               }
               for(int inp =0; inp < in+1; inp++)
               {
                   for(int outp =0; outp < out; outp++)
                   {
                       try {

                       matrix[inp][outp] =  randWeight;
                       }
                       catch (const std::out_of_range& e) {
                           std::cout << "Out of Range error.2\n";;
                           std::cerr << e.what();
                       } catch (const std::exception& e) {
                               std::cout << "Out of Range error.02\n";;
                               std::cerr << e.what();
                       } catch (const std::string& e) {
                               std::cout << "Out of Range error.20\n";;
                               //std::cerr << e.what();
                       } catch (...) {
                               std::cout << "Out of Range error.22\n";;
                       }
//                       std::cout << " - " + std::to_string(inp) + " - " + std::to_string(outp) + " \n ";
                   }
               }
           }
           void toHiddenLayer(float *inputs)
           {
               for(int hid =0; hid < out; hid++)
               {
                   float tmpS = 0.0;
                   for(int inp =0; inp < in; inp++)
                   {
                       tmpS += inputs[inp] * matrix[inp][hid];
                   }
                   tmpS += matrix[in][hid];
                   hidden[hid] = sigmoida(tmpS);
               }
           };
           float* getHidden()
           {
               return hidden;
           };
           float calcOutError(float *targets, bool & showError )
           {
               float errsum = 0.0;
               errors = (float*) malloc((out)*sizeof(float));
               for(int ou =0; ou < out; ou++)
               {
                   float errTmp = (targets[ou] - hidden[ou]) * sigmoidasDerivate(hidden[ou]);

                  /* if (!isnan(errTmp)) std::cout << " - " + std::to_string(errTmp]) + " - " + std::to_string(out) + " \n ";*/
                   //if (errTmp > errLimitG)showError = true;

                   errors[ou] = errTmp;

                   errsum += absF(errTmp);
               }
               return errsum;
           };
           void calcHidError(float *targets,float **outWeights,int inS, int outS, bool & showError)
           {
               errors = (float*) malloc((inS)*sizeof(float));
               for(int hid =0; hid < inS; hid++)
               {
                   errors[hid] = 0.0;
                   for(int ou =0; ou < outS; ou++)
                   {
                       errors[hid] += targets[ou] * outWeights[hid][ou];

                       //if(!isnan(errors[hid])) std::cout << " - " + std::to_string(errors[hid]) + " - " + std::to_string(outS) + " \n ";
                       

                   }

                   //if (errors[hid] > errLimitG)showError = true;

                   errors[hid] *= sigmoidasDerivate(hidden[hid]);
               }
           };
           float* getErrors()
           {
               return errors;
           };
           float sigmoida(float val)
           {
              return (1.0 / (1.0 + exp(-val)));
           }
           float sigmoidasDerivate(float val)
           {
                return (val * (1.0 - val));
           };
    };

    float ** feedForwarding(bool mode_train);
    float ** backPropagate();
    void optimiseWay();
    float* processErrors(int i, bool & startOptimisation, bool showError, float totalE);
    float ** train(float *in, float *targ);
    void query(float *in);
    void printArray(float *arr, int iList, int s);
    float * sumFloatMD(float * left,float *right,int inS);
    int nlCount;
    struct nnLay *list;

private:
    int inputNeurons;
    int outputNeurons;
    float errLimit;
    float errOptinizationLimit;
    float *inputs;
    float *targets;
};

#endif // MYNEURO_H
