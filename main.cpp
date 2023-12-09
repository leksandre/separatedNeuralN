//#include <QCoreApplication>
//#include <QDebug>
//#include <QTime>



//for linux
//#include "myNeuro.cpp"
//#include <sys/time.h>

//for win!!
#include "myNeuro.h"
#include <time.h>
//for openGL visulaise
#include <Windows.h>
#include <GL\glew.h>
//do not uncomment #define FREEGLUT_STATIC
//do not uncomment #define _LIB
//do not uncomment #define FREEGLUT_LIB_PRAGMAS 0
#include <GL\freeglut.h>
#include <iostream>
//do not uncomment #include <windows.h>
//do not uncomment #include <gl\gl.h>






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





















//start openGL visualise
#include <GL\glut.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <conio.h>
#include "iostream"
#include <time.h>
using namespace std;

bool flag = true;
GLfloat xRotated, yRotated, zRotated;

void changeViewPort(int w, int h)
{
    glViewport(0, 0, w, h);
}


void check(int value)
{
    if (flag == true)
    {
        flag = false;
    }
    else if (flag == false)
    {
        flag = true;
    }
    cout << flag << "\n";
    glutPostRedisplay();
    glutTimerFunc(100, check, 0);

}

void highlight_lines(float x, float y, float z, float live_transparency_line)
{
    float a, b, c;
    for (a = 0; a <= 0.9; a = a + 0.1)
    {
        for (b = 0; b <= 0.9; b = b + 0.1)
        {
            for (c = 0; c <= 0.9; c = c + 0.1)
            {
                //First Hidden Layer Plane 1 
                glPointSize(12.0);
                glBegin(GL_POINTS);
                glColor4f(1.0, 1.0, 1.0, 0.05);
                glVertex3f(0.1 + b, 0.1 + c, 0.4);
                glVertex3f(0.1 + b, 0.1 + c, 0.4);
                glEnd();
                glBegin(GL_LINE_LOOP);
                glColor4f(1.0, 1.0, 1.0, live_transparency_line);
                glVertex3f(x, y, z);
                glVertex3f(0.1 + b, 0.1 + c, 0.4);
                glVertex3f(0.1 + b, 0.1 + c, 0.4);
                glEnd();

                //First Hidden Layer Plane 4
                if (c < 0.44)
                {
                    glPointSize(12.0);
                    glBegin(GL_POINTS);
                    glVertex3f(x, y, z);
                    glVertex3f(0.17 + b, 0.16 + c, 0.44);
                    glVertex3f(0.17 + b, 0.16 + c, 0.44);
                    glEnd();
                    glBegin(GL_LINE_LOOP);
                    glColor4f(1.0, 1.0, 1.0, live_transparency_line);
                    glVertex3f(x, y, z);
                    glVertex3f(0.17 + b, 0.16 + c, 0.44);
                    glVertex3f(0.17 + b, 0.16 + c, 0.44);
                    glEnd();
                }

                //Second Hidden Layer Plane 1 
                glPointSize(12.0);
                glBegin(GL_POINTS);
                glColor4f(1.0, 1.0, 1.0, 0.05);
                glVertex3f(0.1 + b, 0.1 + c, 0.8);
                glVertex3f(0.1 + b, 0.1 + c, 0.8);
                glEnd();
                glBegin(GL_LINE_LOOP);
                glColor4f(1.0, 1.0, 1.0, live_transparency_line);
                glVertex3f(0.1 + b, 0.1 + a, 0.4);
                glVertex3f(0.1 + b, 0.1 + c, 0.8);
                glVertex3f(0.1 + b, 0.1 + c, 0.8);
                glEnd();

                //output layer 
                glPointSize(20.0);
                glBegin(GL_POINTS);
                glColor4f(0.0, 0.0, 1.0, 1.0);
                glVertex3f(0.2, 0.35, 1.2);
                glEnd();
                glBegin(GL_LINE_LOOP);
                glColor4f(1.0, 1.0, 1.0, live_transparency_line);
                glVertex3f(0.1 + b, 0.1 + a, 0.8);
                glVertex3f(0.2, 0.35, 1.2);
                glVertex3f(0.2, 0.35, 1.2);
                glEnd();

            }
        }
    }
}
void displaynetwork(void)
{
    glMatrixMode(GL_MODELVIEW);
    // clear the drawing buffer.
    glClear(GL_COLOR_BUFFER_BIT);
    // clear the identity matrix.
    glLoadIdentity();
    // traslate the draw by z = -4.0
    // Note this when you decrease z like -8.0 the drawing will looks far , or smaller.
    glTranslatef(-1.0, -0.75, -3.5); // -1.6 for scaling of 2.3
    // Red color used to draw.
    glColor3f(0.8, 0.2, 0.1);
    glRotatef(yRotated, 1.0, 1.0, 1.0);
    glScalef(1.0, 1.0, 1.0);

    float dead_transparency_line = 0.08;
    float live_transparency_line = 0.15;
    float a, b, c;
    for (a = 0; a <= 0.9; a = a + 0.1)
    {
        for (b = 0; b <= 0.9; b = b + 0.1)
        {
            for (c = 0; c <= 0.9; c = c + 0.1)
            {
                //Input Layer 
                glPointSize(15.0);
                glBegin(GL_POINTS);
                glColor4f(1.0, 1.0, 1.0, 0.05);
                glVertex3f(0.1 + b, 0.1 + a, 0.0);
                glEnd();

                //First Hidden Layer Plane 1 
                glPointSize(12.0);
                glBegin(GL_POINTS);
                glColor4f(1.0, 1.0, 1.0, 0.05);
                glVertex3f(0.1 + b, 0.1 + c, 0.4);
                glEnd();
                glBegin(GL_LINE_LOOP);
                glColor4f(1.0, 1.0, 1.0, dead_transparency_line);
                glVertex3f(0.1 + b, 0.1 + a, 0.0);
                glVertex3f(0.1 + b, 0.1 + c, 0.4);
                glVertex3f(0.1 + b, 0.1 + c, 0.4);
                glEnd();

                //First Hidden Layer Plane 2
                if (c < 0.47)
                {
                    glPointSize(12.0);
                    glBegin(GL_POINTS);
                    glVertex3f(0.1 + b, 0.1 + a, 0.0);
                    glVertex3f(0.13 + b, 0.13 + c, 0.42);
                    glEnd();
                    glBegin(GL_LINE_LOOP);
                    glColor4f(1.0, 1.0, 1.0, dead_transparency_line);
                    glVertex3f(0.1 + b, 0.1 + a, 0.0);
                    glVertex3f(0.13 + b, 0.13 + c, 0.42);
                    glVertex3f(0.13 + b, 0.13 + c, 0.42);
                    glEnd();
                }

                //First Hidden Layer Plane 3
                glPointSize(12.0);
                glBegin(GL_POINTS);
                glVertex3f(0.1 + b, 0.1 + a, 0.0);
                glVertex3f(0.07 + b, 0.07 + c, 0.42);
                glVertex3f(0.07 + b, 0.07 + c, 0.42);
                glEnd();
                glBegin(GL_LINE_LOOP);
                glColor4f(1.0, 1.0, 1.0, dead_transparency_line);
                glVertex3f(0.1 + b, 0.1 + a, 0.0);
                glVertex3f(0.07 + b, 0.07 + c, 0.42);
                glVertex3f(0.07 + b, 0.07 + c, 0.42);
                glEnd();

                //First Hidden Layer Plane 4
                if (c < 0.44)
                {
                    glPointSize(12.0);
                    glBegin(GL_POINTS);
                    glVertex3f(0.1 + b, 0.1 + a, 0.0);
                    glVertex3f(0.17 + b, 0.16 + c, 0.44);
                    glVertex3f(0.17 + b, 0.16 + c, 0.44);
                    glEnd();
                    glBegin(GL_LINE_LOOP);
                    glColor4f(1.0, 1.0, 1.0, dead_transparency_line);
                    glVertex3f(0.1 + b, 0.1 + a, 0.0);
                    glVertex3f(0.17 + b, 0.16 + c, 0.44);
                    glVertex3f(0.17 + b, 0.16 + c, 0.44);
                    glEnd();
                }

                //Second Hidden Layer Plane 1 
                glPointSize(12.0);
                glBegin(GL_POINTS);
                glColor4f(1.0, 1.0, 1.0, 0.05);
                glVertex3f(0.1 + b, 0.1 + c, 0.8);
                glVertex3f(0.1 + b, 0.1 + c, 0.8);
                glEnd();
                glBegin(GL_LINE_LOOP);
                glColor4f(1.0, 1.0, 1.0, dead_transparency_line);
                glVertex3f(0.1 + b, 0.1 + a, 0.4);
                glVertex3f(0.1 + b, 0.1 + c, 0.8);
                glVertex3f(0.1 + b, 0.1 + c, 0.8);
                glEnd();

                //Second Hidden Layer Plane 2
                if (c < 0.47)
                {
                    glPointSize(12.0);
                    glBegin(GL_POINTS);
                    glVertex3f(0.1 + b, 0.1 + a, 0.42);
                    glVertex3f(0.13 + b, 0.13 + c, 0.82);
                    glVertex3f(0.13 + b, 0.13 + c, 0.82);
                    glEnd();
                    glBegin(GL_LINE_LOOP);
                    glColor4f(1.0, 1.0, 1.0, dead_transparency_line);
                    glVertex3f(0.1 + b, 0.1 + a, 0.0);
                    glVertex3f(0.13 + b, 0.13 + c, 0.82);
                    glVertex3f(0.13 + b, 0.13 + c, 0.82);
                    glEnd();
                }

                //Output Layer
                glPointSize(12.0);
                glBegin(GL_POINTS);
                glColor4f(1.0, 1.0, 1.0, 0.05);
                glVertex3f(0.1 + b, 0.35, 1.2);
                glVertex3f(0.1 + b, 0.35, 1.2);
                glEnd();
                glBegin(GL_LINE_LOOP);
                glColor4f(1.0, 1.0, 1.0, dead_transparency_line);
                glVertex3f(0.1 + b, 0.1 + c, 0.8);
                glVertex3f(0.1 + a, 0.35, 1.2);
                glVertex3f(0.1 + a, 0.35, 1.2);
                glEnd();
            }
        }
    }

    if (flag == true)
    {
        //Inut image '1'
        glPointSize(15.0);
        glBegin(GL_POINTS);
        glColor4f(0.0, 0.0, 1.0, 1.0);
        glVertex3f(0.3, 0.2, 0.0);
        glVertex3f(0.3, 0.3, 0.0);
        glVertex3f(0.3, 0.4, 0.0);
        glVertex3f(0.3, 0.5, 0.0);
        glVertex3f(0.3, 0.6, 0.0);
        glVertex3f(0.4, 0.5, 0.0);
        glVertex3f(0.2, 0.2, 0.0);
        glVertex3f(0.3, 0.2, 0.0);
        glVertex3f(0.4, 0.2, 0.0);
        glEnd();

        //Highlighting the active neurons 
        highlight_lines(0.3, 0.2, 0.0, live_transparency_line);
        highlight_lines(0.3, 0.3, 0.0, live_transparency_line);
        highlight_lines(0.3, 0.4, 0.0, live_transparency_line);
        highlight_lines(0.3, 0.5, 0.0, live_transparency_line);
        highlight_lines(0.3, 0.6, 0.0, live_transparency_line);
        highlight_lines(0.4, 0.5, 0.0, live_transparency_line);
        highlight_lines(0.2, 0.2, 0.0, live_transparency_line);
        highlight_lines(0.3, 0.2, 0.0, live_transparency_line);
        highlight_lines(0.4, 0.2, 0.0, live_transparency_line);
    }
    //Flushing the whole output
    glFlush();
    // sawp buffers called because we are using double buffering 
    glutSwapBuffers();
}
void reshapenetwork(int x, int y)
{
    if (y == 0 || x == 0) return;  //Nothing is visible then, so return
    //Set a new projection matrix
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    //Angle of view:40 degrees
    gluPerspective(40.0, (GLdouble)x / (GLdouble)y, 0.5, 20.0);
    glViewport(0, 0, x, y);  //Use the whole window for rendering
}
void idlenetwork(void)
{
    yRotated += 0.05;
    displaynetwork();
}
void render()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glutSwapBuffers();
}




int main(int argc, char *argv[])
{
    //for openGl visulaise
    // 
    //Initialize GLUT
    glutInit(&argc, argv);
    //double buffering used to avoid flickering problem in animation
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
    // window size
    glutInitWindowSize(1350, 950);
    glutInitWindowPosition(0, 0);
    // create the window 
    glutCreateWindow("network Rotating Animation");
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    yRotated = 40;
    glClearColor(0.0, 0.0, 0.0, 0.0);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    //Assign  the function used in events
    glutDisplayFunc(displaynetwork);
    glutReshapeFunc(reshapenetwork);
    glutIdleFunc(idlenetwork);
    //Let start glut loop
    glutTimerFunc(100, check, 0);
    glutMainLoop();
    return 0;











    //for openGl visulaise // empty window
    //
    glutInit(&argc, argv);  // Initialize GLUT
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);   // Set up some memory buffers for our display
    glutInitWindowSize(900, 700);  // Set the window size
    glutCreateWindow("Tactical_Hacker, GL");   // Create the window with the title "Tactical_Hacker,GL"
    glutReshapeFunc(changeViewPort);  // Bind the two functions (above) to respond when necessary
    glutDisplayFunc(render);
    // Very important!  This initializes the entry points in the OpenGL driver so we can call all the functions in the API.
    GLenum err = glewInit();
    if (GLEW_OK != err) {
        fprintf(stderr, "GLEW error");
        return 1;
    }
    glutMainLoop();
    return 0;














    //QCoreApplication a(argc, argv);
//   std::cout<<"\n_________________________________ start main 0\n";;;
    time_t start, end;
    double time_taken;
    time(&start);


    myNeuro* bb = new myNeuro();
    iCycleTotal = 0;
//    start_varian1 = false;
    bool start_varian1 = true;


    if (start_varian1) {






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

        std::cout << "\n________________target1_________________\n";;
        for (int out = 0; out < 10; out++) {
            std::cout << "outputNeuron " + std::to_string(out) + ":";
            std::cout << std::to_string(tar1[out]) + "\n";
        }
        std::cout << "\n________________target2_________________\n";;
        for (int out = 0; out < 10; out++) {
            std::cout << "outputNeuron " + std::to_string(out) + ":";
            std::cout << std::to_string(tar2[out]) + "\n";
        }

        //--------------------------------NN---------WORKING---------------

        std::cout << "\n___________________calculate_without_train_____________\n";;
        bb->query(abc);
        bb->query(cba);

        std::cout << "\n________________start_train_________________\n";;
        iCycle = 0;
        int nTrainingSimple = 100000;




//        float * e1;
//        for(int i=(bb->nlCount-2);i>=0;i--)
//            errTmp[i] = e1;

        while (iCycle < nTrainingSimple and !is_optimizedM)
        //train until specified accuracy level
        {
            if(!is_optimizedM) iCycleTotal++;

            float ** errors1 = bb->train(abc, tar1,allow_optimisation_transform);

//            if(iCycle==0)
//            for(int i=(bb->nlCount-2);i>=0;i--)
//                errTmp[i] = bb->list[i].getErrors();

//            for(int i=(bb->nlCount-2);i>=0;i--)
//                errTmp[i] = bb->sumFloatMD(errTmp[i],bb->list[i].getErrors(),bb->list[i].getOutCount());


            float ** errors2 = bb->train(cba, tar2,allow_optimisation_transform);

//            for(int i=(bb->nlCount-2);i>=0;i--)
//                errTmp[i] = bb->sumFloatMD(errTmp[i],bb->list[i].getErrors(),bb->list[i].getOutCount());


//            for(int i=(bb->nlCount-2);i>=0;i--)
//                errTmp[i] = bb->list[i].getErrors();


//            for(int i=(bb->nlCount-2);i>=0;i--)
//                errTmp[i] = bb->sumFloatMD(errTmp[i],bb->list[i].getErrors(),bb->list[i].getOutCount());

            iCycle++;

//            if(iCycle>1000)
            if (allow_optimisation_transform){
                for(int i=(bb->nlCount-2);i>=0;i--)
                    bb->optimize_layer(i);
            }


            if(is_optimizedM)
            {
                cout<<endl<<"_______________show_errors_______________\n"<<endl;
//                for(int i=(bb->nlCount-1);i>=0;i--) //skip "out" layer (nlCount-1) we could not modified it (a am too stupid for that)
                for(int i=(bb->nlCount-2);i>=0;i--)
                {

                    cout<<" layer :"+std::to_string(i)+" ";
                    bb->printArray(bb->list[i].getErrors(),i, bb->list[i].getOutCount());
                    std::cout<<"\n";

                    cout<<" errTmp:"+std::to_string(i)+" ";
                    bb->printArray(bb->list[i].errTmp,i, bb->list[i].getOutCount());
                    std::cout<<"\n";
//  just trash
//                    bb->printArray(errors1[i],0,300);
//                    bb->printArray(errors2[i],0,300);
//                    for(int j=0;j<241;j++)
//                    {
//                        cout<<"\t"<<i<<":"<<j<<"\t"<<errors1[i][j];
//                        if(errors1[i][j]!=0)maxN=j;
//                    }
//                    std::cout << "\nmaxN="<<maxN;
//                    cout<<endl<<"\n______________________________\n"<<endl;


                }
            }



        }


        std::cout << "\n________________end_train_________________\n";;
        std::cout << "\n___________________calculate_RESULT_____________\n";;
        bb->query(abc);
        std::cout << "______\n";;
        bb->query(cba);


        std::cout << "\n_______________THE____END_______________\n";;
        //std::cout<<"\n_______________THE____END_______________\n";;

         //return a.exec();



    std::cout<<"\n______________________________\n";
    std::cout<<"iCycle:"<<iCycle<<endl;
    std::cout<<"iCycleTotal:"<<iCycleTotal<<endl;

    time(&end);
    // Calculating total time taken by the program.

    time_taken = double(end - start);
    std::cout << "Time taken by program is : " << time_taken << "";
    std::cout << " sec " << "\n";


    if (allow_optimisation_transform) {
        bb->write_matrix_var1(model_fn_opt);
    } else {
        bb->write_matrix_var1(model_fn);
    }


    return 0;

    }














    iCycle = 10;//не показывать в начале лог ошибок
    iCycleTotal = 0;
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
    std::cout << "\n________________start_train_________________\n";;



    while(!is_optimizedM){
        for (int sample = 1; sample <= nTraining; ++sample) {

            if(!is_optimizedM)iCycle++;
            if(is_optimizedM)continue;
            iCycleTotal++;
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
            bb->train(binNumber, target,allow_optimisation_transform);

            if (allow_optimisation_transform){

                for(int i=(bb->nlCount-2);i>=0;i--){

                   // errTmp[i] = bb->sumFloatMD(errTmp[i],bb->list[i].getErrors(),bb->list[i].getOutCount());
                    bb->optimize_layer(i);
                }
            }




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




        //reopen our file with images and labels
        image.clear();
        image.seekg(0);
        label.clear();
        label.seekg(0);
        for (int i = 1; i <= 16; ++i) {
            image.read(&number, sizeof(char));
        }
        for (int i = 1; i <= 8; ++i) {
            label.read(&number, sizeof(char));
        }

    }




    if(is_optimizedM)
    {
        cout<<endl<<"_______________show_errors_______________\n"<<endl;
       for(int i=(bb->nlCount-2);i>=0;i--)
        {

            cout<<" layer :"+std::to_string(i)+" ";
            bb->printArray(bb->list[i].getErrors(),i, bb->list[i].getOutCount());
            std::cout<<"\n";

            if (allow_optimisation_transform) {
                cout << " errTmp:" + std::to_string(i) + " ";
                bb->printArray(bb->list[i].errTmp, i, bb->list[i].getOutCount());
                std::cout << "\n";
            }



        }
    }



    std::cout << "\n________________end_train_________________\n";;

    std::cout << "\n___________________calculate_RESULT_____________\n";;

    for (int sample = 1; sample <= 10; ++sample) {
        ////cout << "Sample " << sample << endl;
        //// Getting (image, label)
        int labelN = input(true);
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

    std::cout<<"\n______________________________\n";
    std::cout<<"iCycle:"<<iCycle<<endl;
    std::cout<<"iCycleTotal:"<<iCycleTotal<<endl;

    // Save the final network
    write_matrix(model_fn);
    report.close();
    image.close();
    label.close();















    time(&end);
    // Calculating total time taken by the program.

    time_taken = double(end - start);
    std::cout << "Time taken by program is : " << time_taken << "";
    std::cout << " sec " << "\n";

    return 0;
}
