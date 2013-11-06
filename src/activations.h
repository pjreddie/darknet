typedef enum{
    SIGMOID, RELU, IDENTITY
}ACTIVATOR_TYPE;

double relu_activation(double x);
double relu_gradient(double x);
double sigmoid_activation(double x);
double sigmoid_gradient(double x);
double identity_activation(double x);
double identity_gradient(double x);
