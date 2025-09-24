#include<iostream>
#include<fstream>
#include<sstream>
#include<string>
#include<cmath>

using namespace std;

const string HISTORICAL_FILE = "historical_data.csv";
const string PREDICTIONS_FILE = "predictions.csv";

const int MAX_LOCATIONS = 50;

///////////////////////////////////////////////////////////////
//
// Structure : House
// Use : Holds information about house data
//
///////////////////////////////////////////////////////////////
struct House
{
    float area;
    int bedroom;
    int bathrooms;
    int age;
    string location;
    float price;
};

///////////////////////////////////////////////////////////////
//
// Class : RegressionModel
//
///////////////////////////////////////////////////////////////
class RegressionModel
{
private:
    House* dataset;
    int nHouses;
    int capacity;
    float* coefficients;
    string locations[MAX_LOCATIONS];
    int nLocations;
    bool trained;

public:
    RegressionModel()
    {
        capacity = 100;
        dataset = new House[capacity];
        nHouses = 0;

        // Allocate a dynamic array of 6 floats (new float[6]).
        // Initialize the first element to 0.
        coefficients = new float[6]{0};
        nLocations = 0;
        trained = false;
    }

    ~RegressionModel()
    {
        delete[] dataset;
        delete[] coefficients;
    }

////////////////////////////////////////////////////////////////////////////////////
//
// Function : LocationScore
// Use : Get location score by index
//
////////////////////////////////////////////////////////////////////////////////////
    int LocationScore(const string& loc)
    {
        for (int i = 0; i < nLocations; i++)
        {
            if (locations[i] == loc)
            {
                return i + 1;
            }
        }

        if (nLocations < MAX_LOCATIONS)
        {
            locations[nLocations] = loc;
            nLocations++;

            return nLocations;
        }
        
        return 1;
    }

////////////////////////////////////////////////////////////////////////////////////
//
// Function : AddHouse
// Use : Use to add house data
//
////////////////////////////////////////////////////////////////////////////////////
    void AddHouse(House h, bool SaveToCSV = false)
    {
        int NewCapacity = 0;
        int i = 0;

        if (nHouses >= capacity)
        {
            NewCapacity =  capacity * 2;
            House* temp = new House[NewCapacity];

            for (i = 0; i < capacity; i++)
            {
                temp[i] = dataset[i];
            }

            delete[] dataset;

            dataset = temp;
            capacity = NewCapacity;
        }
        dataset[nHouses++] = h;

        // Store location if new
        LocationScore(h.location);

        if (SaveToCSV)
        {
            ofstream file(HISTORICAL_FILE, ios::app);

            if (file.is_open())
            {
                file << h.area << "," << h.bedroom << "," << h.bathrooms << ","
                     << h.age <<"," << h.location << "," << h.price <<endl;

                file.close();
            }
        }
    }

////////////////////////////////////////////////////////////////////////////////////
//
// Function : LoadCSV
// Use : Use to load CSV data
//
////////////////////////////////////////////////////////////////////////////////////
    void LoadCSV(const string& filename)
    {
        string line;
        int i = 0;

        ifstream file(filename);
        
        if (!file.is_open())
        {
            cout << "Failed to open file...!\n";
            
            return;
        }

        // Skip header
        getline(file, line);

        while (getline(file, line))
        {
            stringstream ss (line);
            string token, row[6];
            int i = 0;

            while(getline(ss, token, ',') && i < 6)
            {
                row[i++] = token;
            }

            // Skip malformed rows
            if (i < 6)
            {
                continue;
            }

            House h;
            h.area = stof(row[0]);
            h.bedroom = stof(row[1]);
            h.bathrooms = stof(row[2]);
            h.age = stof(row[3]);
            h.location = row[4];
            h.price = stof(row[5]);

            // Do not save again
            AddHouse(h);
        }
        file.close();
        cout << "CSV data loaded successfully...\n";
    }

////////////////////////////////////////////////////////////////////////////////////
//
// Function : DisplayData
// Use : Dsiaply data on the console
//
////////////////////////////////////////////////////////////////////////////////////
    void DisplayData()
    {
        int i = 0;

        if (nHouses == 0)
        {
            cout << "No historical data...\n";

            return;
        }

        cout << "\n----- Historical House Data -----\n";

        for (i = 0; i < nHouses; i++)
        {
            cout << i + 1 << ". Area: " << dataset[i].area
                 << " sq ft, Bedrooms: " << dataset[i].bedroom
                 << ", Bathrooms: " << dataset[i].bathrooms
                 << ", Age: "<< dataset[i].age
                 << ", Location: " << dataset[i].location
                 << ", Price: Rs" <<dataset[i].price
                 <<endl;
        }
    }

////////////////////////////////////////////////////////////////////////////////////
//
// Function : Train
// Use : Train the model to predict price
//
////////////////////////////////////////////////////////////////////////////////////
    void Train()
    {
        int m = 5, i = 0, j = 0, k = 0, t = 0;

        if (nHouses == 0)
        {
            cout << "No data to train...\n";

            return;
        }

        if (nHouses < 6)
        {
            cout << "Need at-least 6 historical houses to train the model...\n";

            return;
        }

        float** X = new float*[nHouses];
        float* Y = new float[nHouses];

        for (i = 0; i < nHouses; i++)
        {
            X[i] = new float[m + 1];

            X[i][0] = 1;
            X[i][1] = dataset[i].area;
            X[i][2] = dataset[i].bedroom;
            X[i][3] = dataset[i].bathrooms;
            X[i][4] = dataset[i].age;
            X[i][5] = LocationScore(dataset[i].location);
            Y[i] = dataset[i].price;
        }

        float** XtX = new float*[m + 1];

        for ( i = 0; i <= m; i++)
        {
            XtX[i] = new float[m+1]{0};
        }

        float XtY[6]={0};

        for( i = 0; i < nHouses; i++)
        {
            for(j = 0; j <= m; j++)
            {
                XtY[j] += X[i][j]*Y[i];
                for(k = 0; k <= m; k++)
                {
                    XtX[j][k] += X[i][j]*X[i][k];
                }
            }
        }

        float aug[6][7];

        for(i = 0; i <= m; i++)
        {
            for(j = 0; j<=m; j++) 
            {
                aug[i][j]=XtX[i][j];
                aug[i][m+1] = XtY[i];
            }
        }

        for(i = 0; i<=m ; i++)
        {
            for(k = i+1; k<= m; k++)
            {
                if(fabs(aug[k][i]) > fabs(aug[i][i]))
                {
                    for(t = 0; t <= m+1; t++)
                    {
                        swap(aug[i][t], aug[k][t]);
                    }
                }
            }

            float diag = aug[i][i];

            if(diag == 0) 
            {
                continue;
            }

            for(j = i;j <= m+1; j++) 
            {
                aug[i][j]/=diag;
            }

            for(k = 0;k <= m; k++)
            {
                if(k!=i)
                {
                    float factor = aug[k][i];

                    for(j = i; j <= m+1; j++) 
                    {
                        aug[k][j] -= factor*aug[i][j];
                    }
                }
            }
        }

        for(i = 0; i <= m; i++) 
        {
            coefficients[i]=aug[i][m+1];
        }
            trained = true;

        cout << "Training completed. Coefficients:\n";

        for(i = 0; i <= m; i++)
        {
            cout << "b" << i << " = " << coefficients[i] << " ";
        }
        cout << endl;

        for(i = 0; i < nHouses; i++)
        {
            delete[] X[i];
        } 
        delete[] X;

        for(i = 0; i <= m; i++)
        {
            delete[] XtX[i];
        } 
        delete[] XtX;
    }

////////////////////////////////////////////////////////////////////////////////////
//
// Function : Predict
// Use : Will predict the price of house
//
////////////////////////////////////////////////////////////////////////////////////
    float Predict(House h, bool SavePrediction = false)
    {
        if (!trained)
        {
            cout << "Model not trained yet... Please train before predicting...\n";

            return 0;
        }
        
        int LocScore = LocationScore(h.location);

        float price = coefficients[0] + 
                      coefficients[1]*h.area + 
                      coefficients[2]*h.bedroom +
                      coefficients[3]*h.bathrooms +
                      coefficients[4]*h.age +
                      coefficients[5]*LocScore;

        if (SavePrediction)
        {
            ofstream file(PREDICTIONS_FILE, ios::app);

            if (file.is_open())
            {
                file << h.area << ","
                     << h.bedroom << ","
                     << h.bathrooms << ","
                     << h.age << ","
                     << h.location << ","
                     << price
                     <<endl;

                file.close();
            }
        }
        return price;
    }
};

////////////////////////////////////////////////////////////////////////////////////
//
// Entry Point Function (main)
//
////////////////////////////////////////////////////////////////////////////////////
int main()
{
    RegressionModel model;
    int iChoice = 0;

    do
    {
        cout << "\n--- House Price Prediction System ---\n";
        cout << "1. Load historical data from CSV\n";
        cout << "2. Add historical data manually\n";
        cout << "3. Display historical data\n";
        cout << "4. Train regression model\n";
        cout << "5. Predict house price\n";
        cout << "0. Exit\n";
        cout << "Enter your choice: ";
        cin >> iChoice;

        cin.ignore();

        if(iChoice == 1)
        {
            string filename;

            cout << "Enter CSV filename: "; 
            getline(cin, filename);

            model.LoadCSV(filename);
        }

        else if (iChoice == 2)
        {
            House h;

            cout << "Enter area (sq ft): "; 
            cin >> h.area;
            cout << "Enter bedrooms: "; 
            cin >> h.bedroom;
            cout << "Enter bathrooms: "; 
            cin >> h.bathrooms;
            cout << "Enter age: "; 
            cin >> h.age;

            cin.ignore();

            cout << "Enter location: "; 
            getline(cin, h.location);
            cout << "Enter price: "; 
            cin >> h.price;

            model.AddHouse(h,true);
        }

        else if (iChoice == 3)
        {
            model.DisplayData();
        }

        else if (iChoice == 4)
        {
            model.Train();
        }

        else if (iChoice == 5)
        {
            House h;

            cout << "Enter area (sq ft): "; 
            cin >> h.area;
            cout << "Enter bedrooms: "; 
            cin >> h.bedroom;
            cout << "Enter bathrooms: "; 
            cin >> h.bathrooms;
            cout << "Enter age: "; 
            cin >> h.age;

            cin.ignore();

            cout << "Enter location: "; 
            getline(cin, h.location);

            long predicted = model.Predict(h,true);

            if(predicted !=0 )
            {
                cout << "Predicted Price: Rs " << predicted << endl;
            }
        }

        else if (iChoice == 0)
        {
            cout << "Exiting program.\n";
            exit(EXIT_SUCCESS);
        }
        
        else
        {
            cout << "Invalid choice! Try again.\n";
            exit(EXIT_FAILURE);
        }
    }while(iChoice != 0);
        

    return 0;

}
