// Memory machine: Train a LSTM model to output PI as much as possible.
// Written by Cheng in 2022.5.7

#include <torch/torch.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <vector>

using std::cout, std::endl;

const int N = 10; // size of alphabet
const int M = 8;  // size of input each time (aka. short-term memory)
const int H = 32; // size of hidden layer
const int L = 1;  // size of stacked-LSTM
const double dropout = 0.2; // LSTM dropout
const int batch_size = 2;   // LSTM update batch
const double lr = 1e-3;     // RMSprop lr
const int max_err = 6;      // number of error before retry

const char* data = "14159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534211706798214808651328230664709384460955058223172535940812848111745028410270193852110555964462294895493038196442881097566593344612847564823378678316527120190914";

struct Net : torch::nn::Module {
    Net():
    lstm(register_module("lstm", torch::nn::LSTM(torch::nn::LSTMOptions(N, H).proj_size(N).num_layers(L).dropout(dropout)))),
    optimizer(parameters(), torch::optim::RMSpropOptions(lr)){
        reset();
        optimizer.zero_grad();
    }
    void reset(){
        input = torch::zeros({M, 1, N});
        for(int i=0; i<M; i++) {
            for(int j=0; j<N; j++) {
                input[i][0][j] = 1;
            }
        }
    }
    char forward(char expect) {
        auto [output, _] = lstm->forward(input);
        auto last = output.index({-1, 0}).softmax(0);
        auto result = last.argmax().item().toInt();
        auto loss = torch::cross_entropy_loss(last, torch::tensor((long)(expect-'0')));
        loss.backward();
        if(++update_step >= batch_size) optimizer.step(), optimizer.zero_grad(), update_step = 0;
        for(int i=0; i<M-1; i++) input[i] = input[i+1];
        (input[M-1] = torch::zeros({1, N}))[0][expect-'0'] = 1;
        return '0'+result;
    }
    torch::nn::LSTM lstm;
    torch::Tensor input;
    torch::optim::RMSprop optimizer;
    int update_step = 0;
};

int main() {
    torch::manual_seed(0);
    std::ofstream fout("./output.txt");
    int target = 64; // for(int target = 32; target < strlen(data); target += 4) {
    Net net;
    cout << "Remembering " << target << " digits." << endl;
    for(int iter = 0;; iter++){
        net.reset();
        int bad_cnt = 0;
        char str[target + 1];
        int i = 0;
        while(i<target){
            char now = str[i] = net.forward(data[i]);
            if(now != data[i++] && ++bad_cnt >= max_err) break;
        }
        str[i] = 0;
        if(strncmp(str, data, target) == 0){
            cout << "result = 3." << str << endl;
            cout << "Success. iter = " << iter << endl;
            break;
        }
        // puts(str);
    }
}
