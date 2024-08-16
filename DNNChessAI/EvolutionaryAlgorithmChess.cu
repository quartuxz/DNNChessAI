#include <iostream>
#include <sstream>
#include <fstream>
#include <map>

#include "RealGameSampler.cuh"
#include "TextUIChess.cuh"
#include "NeuralNetwork.cuh"
#include "MatchMaker.cuh"
#include "NNManager.cuh"
#include "MNISTTest.h"



#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//
int main()
{
    MNISTTest test1(MNIST_DEFAULT_TOPOLOGY, R"(C:\Users\Administrator\Desktop\c++\DNNChessAI\DNNChessAI\MINST\mnist_train.csv)", R"(C:\Users\Administrator\Desktop\c++\DNNChessAI\DNNChessAI\MINST\mnist_test.csv)");

    test1.train(20);

    std::cout << test1.test() << std::endl;

    return 0;
    NNManager NNs;
    int optionSelected = 1;
    size_t usedThreads = 1;
    while (optionSelected != 0) {
        //TODO:
        //option to copy NNs
        //option to change amount of threads simulating NNs
        //option to change randomization parameters
        //possibly add NNs with varying topology across generations.
        //possible to make memory optimizations with pre-allocation and block allocation.(withh all working memory)
        std::cout << "1.) Play NN." << std::endl;
        std::cout << "2.) Load NN with Name." << std::endl;
        std::cout << "3.) Load NNs with Name+_Y to _Z number." << std::endl;
        std::cout << "4.) Generate Y-X random NNs with Name_X to Name_Y, if only one is generated then the name is just Name" << std::endl;
        std::cout << "5.) Do X Training generations simulations among NNs with Name+_Y to _Z number. Provide a naming convention, NNs are saved in order of best to worst, namingConvention+_ordinal." << std::endl;
        std::cout << "6.) Do X Training generations simulations among NNs selected. Provide a naming convention, NNs are saved in order of best to worst, namingConvention+_ordinal." << std::endl;
        std::cout << "7.) Save to files NNs with Name+_Y to _Z number." << std::endl;
        std::cout << "8.) Save to file NN with Name." << std::endl;
        std::cout << "9.) Save to files selected NNs." << std::endl;
        std::cout << "10) Show NN names." << std::endl;
        std::cout << "11.) Configue multithreading. " << std::endl;
        std::cout << "0.) Exit." << std::endl;
        std::cin >> optionSelected;


        auto saveNN = [&](const std::string& finalName) {

            std::stringstream ss;
            const std::string& serializedNN = NNs.getNN(finalName)->serialize();
            ss << finalName << ".txt";
            std::ofstream savefile(ss.str(), std::ios::trunc);
            savefile << serializedNN;
            savefile.close();
        };
        
        auto doMM = [&](const std::vector<NeuralNetwork*> selectedNNs, const std::string &namingConvention, unsigned int generations, bool saveAndOverwrite) {            
            MatchMaker mm(selectedNNs);
            mm.setMaxThreads(usedThreads);
            for (size_t i = 0; i < generations; i++) {
                mm.matchMake();

                std::cout << std::endl << i+1 << " generations completed!" << std::endl;
                auto thisGenRes = mm.getNNs();
                if(saveAndOverwrite){
                    for (size_t o = 0; o < thisGenRes.size(); o++)
                    {
                        std::stringstream ss;
                        const std::string& serializedNN = thisGenRes[o]->serialize();
                        ss << namingConvention << "_" << o << ".txt";
                        std::ofstream savefile(ss.str(), std::ios::trunc);
                        savefile << serializedNN;
                        savefile.close();
                    }
                }
            }

            return;
            auto result = mm.getNNs();

            for (size_t i = 0; i < result.size(); i++)
            {
                std::stringstream finalName;
                finalName << namingConvention << "_" << i;
                NNs.addNN(finalName.str(), new NeuralNetwork(*result[i]));
            }
        };

        auto selectNNRange = [&](unsigned int rangeStart, unsigned int rangeEnd, const std::string& name, std::vector<NeuralNetwork*> &selectedNNs ) {            
            for (size_t i = rangeStart; i <= rangeEnd; i++)
            {
                std::stringstream ss;
                ss << name << "_" << i;
                selectedNNs.push_back(NNs.getNN(ss.str()));
            }
        };


        auto selectNNsIndividually = [&](std::vector<NeuralNetwork*> &selectedNNs, std::vector<std::string> *selectedNNsNames = nullptr) {

            std::string option;
            while (true) {
                int optionToInt = 0;
                std::cout << NNs.showNNs();
                std::cout << "enter option or stop: ";
                std::cin >> option;
                if (option == "stop") {
                    break;
                }
                optionToInt = std::atoi(option.c_str());
                selectedNNs.push_back(NNs.getNN(optionToInt));
                if (selectedNNsNames != nullptr) {
                    selectedNNsNames->push_back(NNs.getNNName(optionToInt));
                }
            }
        };

        try{
        switch (optionSelected)
        {
        case 1:
        {
            std::string NNname;
            std::cout << "enter NN name to play: ";
            std::cin >> NNname;
            std::cout << std::endl;
            TextUIChess uiChess(NNs.getNN(NNname), player::white);
            while (true) {
                uiChess.showBoard();
                uiChess.showMoves();
                if (uiChess.getGameCondition() != gameCondition::playing) {
                    std::cout << getGameConditionString(uiChess.getGameCondition()) << std::endl;
                    break;
                }
                if (uiChess.promptMove() != TextUIChess::promptMoveResullt::good) {
                    break;
                }
            }
        }
        break;
        case 2:
        {
            std::stringstream filename;
            std::string NNname;
            std::cout << "enter NN name to load: ";
            std::cin >> NNname;
            filename << NNname << ".txt";

            
            std::ifstream NNFile;
            NNFile.open(filename.str());
            
            

            std::stringstream buffer;
            buffer << NNFile.rdbuf();

            

            NNs.addNN(NNname, buffer.str());

            std::cout << std::endl << NNname << " was read successfully!";
            NNFile.close();
        }
            break;
        case 3:
        {
            std::string NNName;
            unsigned int rangeStart, rangeEnd;
            std::cout << "enter NN name to load: ";
            std::cin >> NNName;
            std::cout << "enter first number in range: ";
            std::cin >> rangeStart;
            std::cout << "enter second number in range: ";
            std::cin >> rangeEnd;


            for (size_t i = rangeStart; i <= rangeEnd; i++)
            {
                std::stringstream finalNNName;
                finalNNName << NNName << "_" << i;

                std::stringstream filename;

                filename << finalNNName.str() << ".txt";

                std::ifstream NNFile;
                NNFile.open(filename.str());
                std::stringstream buffer;
                buffer << NNFile.rdbuf();

                NNs.addNN(finalNNName.str(), buffer.str());

                NNFile.close();
            }

        }
            break;
        case 4:
        {
            std::string NNName;
            unsigned int numberStart = 0;
            unsigned int numberEnd = 0;
            
            std::cout << "enter NN Name: ";
            std::cin >> NNName;

            std::cout << "enter first range for NNs numbering: ";
            std::cin >> numberStart;

            std::cout << "enter last range for NNs numbering: ";
            std::cin >> numberEnd;

            if (numberStart == numberEnd) {
                NNs.addNN(NNName, new NeuralNetwork(QEAC_DEFAULT_TOPOLOGY,NNInitialization(),LearningSchedule()));
            }
            else {
                for (size_t i = numberStart; i <= numberEnd; i++)
                {
                    std::stringstream finalName;
                    finalName << NNName << "_" << i;
                    NNs.addNN(finalName.str(), new NeuralNetwork(QEAC_DEFAULT_TOPOLOGY, NNInitialization(), LearningSchedule()));
                }
            }
        }
            break;
        case 5:
        {
            std::vector<NeuralNetwork*> selectedNNs;
            unsigned int generations, rangeStart, rangeEnd;
            std::string name, namingConvention, verboseAnswer, saveAndOverwriteStr;
            bool saveAndOverwrite = false;
            std::cout << "enter generations: ";
            std::cin >> generations;
            std::cout << std::endl << "enter name: ";
            std::cin >> name;
            std::cout << std::endl << "enter start of range: ";
            std::cin >> rangeStart;
            std::cout << std::endl << "enter end of range: ";
            std::cin >> rangeEnd;
            std::cout << std::endl << "enter naming convention: ";
            std::cin >> namingConvention;
            std::cout << "save and overwrite after every generation(yes/no)? ";
            std::cin >> saveAndOverwriteStr;

            selectNNRange(rangeStart, rangeEnd, name, selectedNNs);

            doMM(selectedNNs,namingConvention,generations, (saveAndOverwriteStr == "yes" ? true : false));
        }
            break;
        case 6:
        {
            std::vector<NeuralNetwork*> selectedNNs;
            unsigned int generations;
            std::string name, namingConvention, saveAndOverwriteStr;
            std::cout << "enter generations: ";
            std::cin >> generations;
            std::cout << "enter naming convention: ";
            std::cin >> namingConvention;
            std::cout << "save and overwrite after every generation(yes/no)? ";
            std::cin >> saveAndOverwriteStr;

            selectNNsIndividually(selectedNNs);

            doMM(selectedNNs, namingConvention, generations, saveAndOverwriteStr == "yes" ? true : false);
        }
        break;
        case 7:
        {
            std::vector<NeuralNetwork*> selectedNNs;
            unsigned int rangeStart, rangeEnd;
            std::string name;
            std::cout << std::endl << "enter name: ";
            std::cin >> name;
            std::cout << std::endl << "enter start of range: ";
            std::cin >> rangeStart;
            std::cout << std::endl << "enter end of range: ";
            std::cin >> rangeEnd;
            selectNNRange(rangeStart,rangeEnd,name,selectedNNs);
            
            for (size_t i = rangeStart; i <= rangeEnd; i++)
            {
                std::stringstream finalName;
                finalName << name << "_" << i;
                saveNN(finalName.str());
            }
        }
        break;
        case 8:
        {
            std::string name;
            std::cout << "enter name: ";
            std::cin >> name;
            saveNN(name);
        }
        break;
        case 9:
        {
            std::vector<NeuralNetwork*> selectedNNs;
            std::vector<std::string> selectNNNames;
            selectNNsIndividually(selectedNNs);
        }
        break;
        case 10:
        {
            std::cout << NNs.showNNs() << std::endl;
            break;
        }
        case 11:
        {
            int newAmount = -1;
            std::cout << "amount of threads used: " << usedThreads << std::endl;
            std::cout << "new amount(<= 0 to exit): ";
            std::cin >> newAmount;

            if (newAmount > 0) {
                usedThreads = newAmount;
            }
            std::cout << std::endl;
            break;
        }
        default:
            break;
        }
        }
        catch (const NNFindError &err) {
            std::cout << err.what() << std::endl;
            continue;
        }
        catch (const std::exception &err) {
            std::cout << err.what() << std::endl;
            system("pause");
            throw;
        }
    }


    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}
