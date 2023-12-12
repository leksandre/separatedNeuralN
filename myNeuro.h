#ifndef MY_FUnC_HEADER_
#define MY_FUnC_HEADER_

extern bool is_optimizedM;

//switcher
extern bool allow_optimisation_transform;
//bool allow_optimisation_transform = false;
extern double areaWidth;
extern double areaHeight;
extern bool start_visualisation;
extern int iCycle;
extern int iCycleTotal;

//static
static double absD(double N) {
    if(N<0)N=N*-1;
    return N;
}
//static
static float absF(float N) {
    if(N<0)N=N*-1;
    return N;
}
#endif

struct Point {
    double x;
    double y;
};

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
#include <vector>
#include <set>
#include <iterator>
#include <algorithm>


using namespace std;

#include <iostream>
#include <vector>


//#include <cmath>// does not need

// Training image file name
//const string training_image_fn = "C:\\mnist\\train-images.idx3-ubyte";//win
const string training_image_fn = "mnist/train-images.idx3-ubyte";//lin

// Training label file name
//const string training_label_fn = "C:\\mnist\\train-labels.idx1-ubyte";//win
const string training_label_fn = "mnist/train-labels.idx1-ubyte";//lin

// Weights file name
const string model_fn = "model-neural-network.dat";
const string model_fn_opt = "model-neural-network-optimized.dat";

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

const float errLimitG = 0.005;

const float errOptinizationLimitG = 0.001; //0.00003; //0.000001; //0.00003;


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



    struct nnLay{
           int in;
           int out;

           bool is_optimizedL;
           float * errTmp;
           struct Point *  pointsIn_ ;
           struct Point *  pointsOut_ ;
           float ** matrix;
           float * hidden;
           float * errors;
//           int setInCount(int inputs){in=inputs};
//           int setOutCount(int outputs){out=outputs};
            Point * getInPoints(){return pointsIn_;}
            Point * getOutPoints(){return pointsOut_;}
           int getInCount(){return in;}
           int getOutCount(){return out;}
           float **getMatrix(){return matrix;}
           float *getErrorsM(){return errors;}

        void truncMatrixOut(int index) {
               if (out<=index)return;
               if (out<=1)return;
            if(index==0){
                index=1;
            }
            for (int ou = (index - 1); ou < (out - 1); ou++) {
                for (int hid = 0; hid < in; hid++) {
                    matrix[hid][ou] = matrix[hid][ou + 1];
                }
            }
            out = out - 1;
        };

        void truncMatrixIn(int index) {
            if (in<=index)return;
            if (in<=1)return;
            if(index==0){
                index=1;
            }
            for (int ou = 0; ou < out; ou++) {
                for (int hid = index - 1; hid < (in - 1); hid++) {
                    matrix[hid][ou] = matrix[hid + 1][ou];
                }
            }
            in = in - 1;
        };

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
            std::vector<Point> distributePointsEvenly(int numPoints) {
                std::vector<Point> points;
                int numRows = sqrt(numPoints);
                int numCols = ceil(static_cast<double>(numPoints) / numRows);
                double deltaX = areaWidth / (numCols + 1);
                double deltaY = areaHeight / (numRows + 1);
                double startX = deltaX;
                double startY = deltaY;

                for (int row = 0; row < numRows; ++row) {
                    for (int col = 0; col < numCols; ++col) {
                        if (points.size() < numPoints) {
                            Point point;
                            point.x =( startX + col * deltaX )/ areaWidth;
                            point.y =( startY + row * deltaY )/ areaHeight;
                            points.push_back(point);
                        }
                    }
                }

                return points;
            };
               void setIO(int inputs, int outputs)
           {
               in=inputs;
               out=outputs;

               std::cout << " in- " << fixed << std::to_string(in);
               std::cout << " out- " << fixed << std::to_string(out) + " \n ";
               std::cout << " randWeight " + std::to_string(randWeight) + " \n ";

               //    point = (Point*) malloc((nlCount)*sizeof(Point));

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

               errTmp = (float *) malloc((out) * sizeof(float));
               for (int i = 0; i < out; i++)
               { errTmp[i] = 0; };

               std::vector<Point> pointsIn = {};
               std::vector<Point> pointsOut = {};

               pointsIn = distributePointsEvenly(inputs, areaWidth, areaHeight);
               std::cout << "in:" << std::endl;
               pointsIn_ = (Point*)malloc((inputs) * sizeof(Point));

               int iPoints = 0;
               for (const auto& point : pointsIn) {
                   std::cout << "in X: " << point.x << ", Y: " << point.y << std::endl;
                   Point point_;
                   point_.x = point.x;
                   point_.y = point.y;
                   pointsIn_[iPoints] = (point_);
                   iPoints++;
               }

               pointsOut = distributePointsEvenly(outputs, areaWidth, areaHeight);
               std::cout << "out:" << std::endl;
               pointsOut_ = (Point*)malloc((outputs) * sizeof(Point));

               iPoints = 0;
               for (const auto& point : pointsOut) {
                   std::cout << "out X: " << point.x << ", Y: " << point.y << std::endl;
                   Point point_;
                   point_.x = point.x;
                   point_.y = point.y;
                   pointsOut_[iPoints] = (point_);
                   iPoints++;
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

//        float absFh(float N) {
//            if(N<0)N=N*-1;
//            return N;
//        };
//        double absDh(double N) {
//            if(N<0)N=N*-1;
//            return N;
//        }

        float calcOutError(float *targets, bool & showError )
           {
               float errsum = 0.0;
               errors = (float*) malloc((out)*sizeof(float)*4);
               for(int ou =0; ou < out; ou++)
               {
                   float eTmp = (targets[ou] - hidden[ou]) * sigmoidasDerivate(hidden[ou]);

                  /* if (!isnan(eTmp)) std::cout << " - " + std::to_string(eTmp]) + " - " + std::to_string(out) + " \n ";*/
                   //if (eTmp > errLimitG)showError = true;

                   errors[ou] = eTmp;

                   errsum += absF(eTmp);
               }
               return errsum;
           };
           void calcHidError(float *targets,float **outWeights,int inS, int outS, bool & showError)
           {
               errors = (float*) malloc((inS)*sizeof(float)*4);
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
    float ** train(float *in, float *targ, bool optimize);
    void query(float *in);
    void printArray(float *arr, int iList, int s);
    void write_matrix_var1(string file_name);
    void optimize_layer(int i);
    void sumFloatMD(int inS);
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
