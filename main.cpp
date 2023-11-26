//#include <QCoreApplication>
//#include <QDebug>
//#include <QTime>


//for linux
//#include "myNeuro.cpp"
//#include <sys/time.h>

//for win!!
#include "myNeuro.h"
#include <time.h>



ifstream image;
ifstream label;
ofstream report;




const int epochs = 512;
const double learning_rate = 1e-3;
const double momentum = 0.9;
const double epsilon = 1e-3;

// From layer 1 to layer 2. Or: Input layer - Hidden layer
double* w1[n1 + 1], * delta1[n1 + 1], * out1;

// From layer 2 to layer 3. Or; Hidden layer - Output layer
double* w2[n2 + 1], * delta2[n2 + 1], * in2, * out2, * theta2;

// Layer 3 - Output layer
double* in3, * out3, * theta3;
double expected[n3 + 1];

// Image. In MNIST: 28x28 gray scale images.
int d[width + 1][height + 1];


double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}


void init_array() {
    // Layer 1 - Layer 2 = Input layer - Hidden layer
    for (int i = 1; i <= n1; ++i) {
        w1[i] = new double[n2 + 1];
        delta1[i] = new double[n2 + 1];
    }

    out1 = new double[n1 + 1];

    // Layer 2 - Layer 3 = Hidden layer - Output layer
    for (int i = 1; i <= n2; ++i) {
        w2[i] = new double[n3 + 1];
        delta2[i] = new double[n3 + 1];
    }

    in2 = new double[n2 + 1];
    out2 = new double[n2 + 1];
    theta2 = new double[n2 + 1];

    // Layer 3 - Output layer
    in3 = new double[n3 + 1];
    out3 = new double[n3 + 1];
    theta3 = new double[n3 + 1];

    // Initialization for weights from Input layer to Hidden layer
    for (int i = 1; i <= n1; ++i) {
        for (int j = 1; j <= n2; ++j) {
            int sign = rand() % 2;

            // Another strategy to randomize the weights - quite good 
            // w1[i][j] = (double)(rand() % 10 + 1) / (10 * n2);

            w1[i][j] = (double)(rand() % 6) / 10.0;
            if (sign == 1) {
                w1[i][j] = -w1[i][j];
            }
        }
    }

    // Initialization for weights from Hidden layer to Output layer
    for (int i = 1; i <= n2; ++i) {
        for (int j = 1; j <= n3; ++j) {
            int sign = rand() % 2;

            // Another strategy to randomize the weights - quite good 
            // w2[i][j] = (double)(rand() % 6) / 10.0;

            w2[i][j] = (double)(rand() % 10 + 1) / (10.0 * n3);
            if (sign == 1) {
                w2[i][j] = -w2[i][j];
            }
        }
    }
}


void back_propagation() {
    double sum;

    for (int i = 1; i <= n3; ++i) {
        theta3[i] = out3[i] * (1 - out3[i]) * (expected[i] - out3[i]);
    }

    for (int i = 1; i <= n2; ++i) {
        sum = 0.0;
        for (int j = 1; j <= n3; ++j) {
            sum += w2[i][j] * theta3[j];
        }
        theta2[i] = out2[i] * (1 - out2[i]) * sum;
    }

    for (int i = 1; i <= n2; ++i) {
        for (int j = 1; j <= n3; ++j) {
            delta2[i][j] = (learning_rate * theta3[j] * out2[i]) + (momentum * delta2[i][j]);
            w2[i][j] += delta2[i][j];
        }
    }

    for (int i = 1; i <= n1; ++i) {
        for (int j = 1; j <= n2; j++) {
            delta1[i][j] = (learning_rate * theta2[j] * out1[i]) + (momentum * delta1[i][j]);
            w1[i][j] += delta1[i][j];
        }
    }
}


void perceptron() {
    for (int i = 1; i <= n2; ++i) {
        in2[i] = 0.0;
    }

    for (int i = 1; i <= n3; ++i) {
        in3[i] = 0.0;
    }

    for (int i = 1; i <= n1; ++i) {
        for (int j = 1; j <= n2; ++j) {
            in2[j] += out1[i] * w1[i][j];
        }
    }

    for (int i = 1; i <= n2; ++i) {
        out2[i] = sigmoid(in2[i]);
    }

    for (int i = 1; i <= n2; ++i) {
        for (int j = 1; j <= n3; ++j) {
            in3[j] += out2[i] * w2[i][j];
        }
    }

    for (int i = 1; i <= n3; ++i) {
        out3[i] = sigmoid(in3[i]);
    }
}




double square_error() {
    double res = 0.0;
    for (int i = 1; i <= n3; ++i) {
        res += (out3[i] - expected[i]) * (out3[i] - expected[i]);
    }
    res *= 0.5;
    return res;
}

int learning_process() {
    for (int i = 1; i <= n1; ++i) {
        for (int j = 1; j <= n2; ++j) {
            delta1[i][j] = 0.0;
        }
    }

    for (int i = 1; i <= n2; ++i) {
        for (int j = 1; j <= n3; ++j) {
            delta2[i][j] = 0.0;
        }
    }

    for (int i = 1; i <= epochs; ++i) {
        perceptron();
        back_propagation();
        if (square_error() < epsilon) {
            return i;
        }
    }
    return epochs;
}

int input(bool showHitn = true) {
    // Reading image
    char number;
    for (int j = 1; j <= height; ++j) {
        for (int i = 1; i <= width; ++i) {
            image.read(&number, sizeof(char));
            if (number == 0) {
                d[i][j] = 0;
            }
            else {
                d[i][j] = 1;
            }
        }
    }
    if(showHitn) cout << "Image:" << endl;
    for (int j = 1; j <= height; ++j) {
        for (int i = 1; i <= width; ++i) {
            if (showHitn)  cout << d[i][j];
        }
        if (showHitn) cout << endl;
    }

    for (int j = 1; j <= height; ++j) {
        for (int i = 1; i <= width; ++i) {
            int pos = i + (j - 1) * width;
            out1[pos] = d[i][j];
        }
    }

    // Reading label
    label.read(&number, sizeof(char));
    for (int i = 1; i <= n3; ++i) {
        expected[i] = 0.0;
    }
    expected[number + 1] = 1.0;
    if (showHitn) cout << "Label: " << (int)(number) << endl;
    return (int)(number);
}


void write_matrix(string file_name) {
    ofstream file(file_name.c_str(), ios::out);

    // Input layer - Hidden layer
    for (int i = 1; i <= n1; ++i) {
        for (int j = 1; j <= n2; ++j) {
            file << w1[i][j] << " ";
        }
        file << endl;
    }

    // Hidden layer - Output layer
    for (int i = 1; i <= n2; ++i) {
        for (int j = 1; j <= n3; ++j) {
            file << w2[i][j] << " ";
        }
        file << endl;
    }

    file.close();
}


int iCycle;
int main(int argc, char *argv[])
{
    //QCoreApplication a(argc, argv);
//   std::cout<<"\n_________________________________ start main 0\n";;;
    time_t start, end;
    time(&start);

   



    if (false) {

        myNeuro* bb = new myNeuro();


        //    return 0;
        //myNeuro bb;
             //----------------------------------INPUTS----GENERATOR-------------
        //   std::cout<<"\n_________________________________ start main\n";;;
                //qsrand((QTime::currentTime().second()));
        float* abc = new float[n1];
        for (int i = 0; i < n1; i++)
        {
            abc[i] = (rand() % 98) * 0.01 + 0.01;
        }

        float* cba = new float[n1];
        for (int i = 0; i < n1; i++)
        {
            cba[i] = (rand() % 98) * 0.01 + 0.01;
        }

        //---------------------------------TARGETS----GENERATOR-------------
        std::cout << "\n________________TARGETS----GENERATOR_________________\n";;
        float* tar1 = new float[10];
        tar1[0] = 0.01;
        tar1[1] = 0.99;
        tar1[2] = 0;
        tar1[3] = 0;
        tar1[4] = 0;
        tar1[5] = 0;
        tar1[6] = 0;
        tar1[7] = 0;
        tar1[8] = 0;
        tar1[9] = 0;

        float* tar2 = new float[10];
        tar2[0] = 0.99;
        tar2[1] = 0.01;
        tar1[2] = 0;
        tar1[3] = 0;
        tar1[4] = 0;
        tar1[5] = 0;
        tar1[6] = 0;
        tar1[7] = 0;
        tar1[8] = 0;
        tar1[9] = 0;


        //--------------------------------NN---------WORKING---------------

        std::cout << "\n___________________calculate_without_train_____________\n";;
        bb->query(abc);
        bb->query(cba);

        std::cout << "\n________________start_train_________________\n";;
        iCycle = 0;
        while (iCycle < nTraining)
        {
            bb->train(abc, tar1);
            bb->train(cba, tar2);
            iCycle++;
        }

        std::cout << "\n___________________calculate_RESULT_____________\n";;
        bb->query(abc);
        std::cout << "______\n";;
        bb->query(cba);


        std::cout << "\n_______________THE____END_______________\n";;
        //std::cout<<"\n_______________THE____END_______________\n";;

         //return a.exec();

    }


    //return 0;















    report.open(report_fn.c_str(), ios::out);
    image.open(training_image_fn.c_str(), ios::in | ios::binary); // Binary image file
    label.open(training_label_fn.c_str(), ios::in | ios::binary); // Binary label file


    // Reading file headers
    char number;
    for (int i = 1; i <= 16; ++i) {
        image.read(&number, sizeof(char));
    }
    for (int i = 1; i <= 8; ++i) {
        label.read(&number, sizeof(char));
    }

    init_array();


    for (int sample = 1; sample <= nTraining; ++sample) {



        ////cout << "Sample " << sample << endl;
        //// Getting (image, label)
        int labelN = input(false);
        //cout << "labelN " << labelN << endl;
        //// Learning process: Perceptron (Forward procedure) - Back propagation
        //int nIterations = learning_process();


       

        float* binNumber = new float[n1];
        for (int i = 0; i < n1; i++)
        {
            binNumber[i] =out1[i];
        
        }

       

        float* target = new float[10];
        target[labelN] = 1;
        bb->train(binNumber, target);






        //// Write down the squared error
        //cout << "No. iterations: " << nIterations << endl;
        //printf("Error: %0.6lf\n\n", square_error());
        //report << "Sample " << sample << ": No. iterations = " << nIterations << ", Error = " << square_error() << endl;
        //// Save the current network (weights)
        //if (sample % 100 == 0) {
        //    cout << "Saving the network to " << model_fn << " file." << endl;
        //    write_matrix(model_fn);
        //}


    }



    for (int sample = 1; sample <= nTraining; ++sample) {

        ////cout << "Sample " << sample << endl;
        //// Getting (image, label)
        int labelN = input(false);
        cout << "labelN " << labelN << endl;
        //// Learning process: Perceptron (Forward procedure) - Back propagation
        //int nIterations = learning_process();

        float* binNumber = new float[n1];
        for (int i = 0; i < n1; i++)
        {
            binNumber[i] = out1[i];
        }
        bb->query(binNumber);
    }


    // Save the final network
    write_matrix(model_fn);

    report.close();
    image.close();
    label.close();















    time(&end);

    // Calculating total time taken by the program.
    double time_taken;
    time_taken = double(end - start);
    std::cout << "Time taken by program is : " << time_taken << "";
    std::cout << " sec " << "\n";

    return 0;
}
