#ifndef MYNEURO_H
#define MYNEURO_H
#include <iostream>
#include <math.h>
//#include <sdk_dev/math.h>
//#include <QtGlobal>
//#include <QDebug>





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





#define learnRate 0.1
#define randWeight (( ((float)rand() / (float)RAND_MAX) - 0.5)* pow(out,-0.5))
class myNeuro
{
public:
    myNeuro();

    bool couldoptimizeM;

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


               hidden = (float*) malloc((out)*sizeof(float));//*2 malloc fail in counting mem

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
           void calcOutError(float *targets)
           {
               errors = (float*) malloc((out)*sizeof(float));
               for(int ou =0; ou < out; ou++)
               {
                   errors[ou] = (targets[ou] - hidden[ou]) * sigmoidasDerivate(hidden[ou]);
               }
           };
           void calcHidError(float *targets,float **outWeights,int inS, int outS)
           {
               errors = (float*) malloc((inS)*sizeof(float));
               for(int hid =0; hid < inS; hid++)
               {
                   errors[hid] = 0.0;
                   for(int ou =0; ou < outS; ou++)
                   {
                       errors[hid] += targets[ou] * outWeights[hid][ou];
                   }
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

    void feedForwarding(bool mode_train);
    void backPropagate();
    void optimiseWay();
    void processErrors(int i, bool & startOptimisation, bool showError);
    void train(float *in, float *targ);
    void query(float *in);
    void printArray(float *arr, int iList, int s);

private:
    struct nnLay *list;
    int inputNeurons;
    int outputNeurons;
    int nlCount;
    float errLimit;
    float errOptinizationLimit;
    float *inputs;
    float *targets;
};

#endif // MYNEURO_H
