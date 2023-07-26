from django.shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.svm import SVR
import seaborn as sns
from  sklearn.metrics import mean_squared_error
# Create your views here.
def index(request):
    return render(request,'index.html')

def square(request):
   return render(request,'square.html')

def squareresult(request):
     if request.method == 'GET':
        area = int(request.GET.get('area', 0))
        shape = int(request.GET.get('span', 0))
        storey = int(request.GET.get('storey', 0))
        # res = prediction(area, shape, noofguids)
        
        squareres=Squareprediction(area,shape,storey)
      
        context = { 
                    
                   'squareres':squareres
                   }
     
        return render(request, 'squareresult.html', context)
def Squareprediction(area,span,storey):
    dataset = pd.read_csv(r"C:/Users/Atees/Desktop/djangoprojects/ml/construction/static/square.csv")
    def convert_to_int(value):
        numbers = value.split('x')  # Use 'x' as the separator
        integer_values = []
        for num in numbers:
            try:
                num_int = int(num)
                integer_values.append(num_int)
            except ValueError:
                pass  # Skip non-numeric values
        return integer_values

# Load the dataset into a Pandas DataFrame


# Split the column into separate integer columns
    dataset[['Outer height', 'Outer width']] = dataset['Outer column(o/p)'].apply(convert_to_int).apply(pd.Series)

# Drop the original column if needed
    dataset.drop('Outer column(o/p)', axis=1, inplace=True)
    def convert_to_int(value):
        numbers = value.split('x')  # Use 'x' as the separator
        integer_values = []
        for num in numbers:
            try:
                num_int = int(num)
                integer_values.append(num_int)
            except ValueError:
                pass  # Skip non-numeric values
        return integer_values

# Load the dataset into a Pandas DataFrame


# Split the column into separate integer columns
    dataset[['Corner height', 'Corner width']] = dataset['Corner column(o/p)'].apply(convert_to_int).apply(pd.Series)

# Drop the original column if needed
    dataset.drop('Corner column(o/p)', axis=1, inplace=True)
    def convert_to_int(value):
        numbers = value.split('x')  # Use 'x' as the separator
        integer_values = []
        for num in numbers:
            try:
                num_int = int(num)
                integer_values.append(num_int)
            except ValueError:
                pass  # Skip non-numeric values
        return integer_values


    # Split the column into separate integer columns
    dataset[['Inner height', 'Inner width']] = dataset['Inner Column(o/p)'].apply(convert_to_int).apply(pd.Series)

    # Drop the original column if needed
    dataset.drop('Inner Column(o/p)', axis=1, inplace=True)
    def convert_to_int(value):
        numbers = value.split('x')  # Use 'x' as the separator
        integer_values = []
        for num in numbers:
            try:
                num_int = int(num)
                integer_values.append(num_int)
            except ValueError:
                pass  # Skip non-numeric values
        return integer_values


    # Split the column into separate integer columns
    dataset[['Beam height', 'Beam width']] = dataset['Beam(o/p)'].apply(convert_to_int).apply(pd.Series)

    # Drop the original column if needed
    dataset.drop('Beam(o/p)', axis=1, inplace=True)

    dataset.fillna(0, inplace=True)

    # Split the dataset into input features and target variables
    X = dataset[['Span(i/p)', 'Storey(i/p)', 'Area(sqm)(i/p)']]
    y = dataset[['Inner height', 'Inner width', 'Outer height', 'Outer width', 'Corner height', 'Corner width', 'Beam height', 'Beam width']]

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate the mean squared error (MSE)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    # Predict dimensions for new data
    new_data = {
        'Span(i/p)': [span],
        'Storey(i/p)': [storey],
        'Area(sqm)(i/p)': [area],
    }
    new_df = pd.DataFrame(new_data)

    predicted_dimensions = model.predict(new_df)
    inner_dim = predicted_dimensions[0][:2].astype(int)
    outer_dim = predicted_dimensions[0][2:4].astype(int)
    corner_dim = predicted_dimensions[0][4:6].astype(int)
    beam_dim = predicted_dimensions[0][6:].astype(int)

    
    print("Square shape:-")
    print(f"Inner Dimension: {max(inner_dim[0], 250)}x{max(inner_dim[1], 250)}")
    print(f"Outer Dimension: {max(outer_dim[0], 250)}x{max(outer_dim[1], 250)}")
    print(f"Corner Dimension: {max(corner_dim[0], 250)}x{max(corner_dim[1], 250)}")
    print(f"Beam Dimension: {max(beam_dim[0], 250)}x{max(beam_dim[1], 250)}")
    value={
         'inner_dim':f"{max(inner_dim[0], 250)}x{max(inner_dim[1], 300)}",
          'outer_dim':f"{max(outer_dim[0], 250)}x{max(outer_dim[1], 300)}",
          'corner_dim':f"{max(corner_dim[0], 250)}x{max(corner_dim[1], 300)}",
          'beam_dim':f"{max(beam_dim[0], 250)}x{max(beam_dim[1], 250)}",
    }
    return value


def Lshape(request):
    return render(request,'Lshape.html')

def Lshapeprediction(area,span,storey):
    dataset = pd.read_csv(r"C:/Users/Atees/Desktop/djangoprojects/ml/construction/static/L.csv")
    def convert_to_int(value):
        numbers = value.split('X')  # Use 'X' as the separator (capital 'X')
        integer_values = []
        for num in numbers:
            try:
                num_int = int(num)
                integer_values.append(num_int)
            except ValueError:
                pass  # Skip non-numeric values
        return integer_values

    # Load the dataset into a Pandas DataFrame
    data = dataset

    dataset = pd.DataFrame(data)

    # Split the column into separate integer columns
    dataset[['Outer height', 'Outer width']] = dataset['Outer column(o/p)'].apply(convert_to_int).apply(pd.Series)

    # Drop the original column if needed
    dataset.drop('Outer column(o/p)', axis=1, inplace=True)


    def convert_to_int(value):
        numbers = value.split('X')  # Use 'X' as the separator (capital 'X')
        integer_values = []
        for num in numbers:
            try:
                num_int = int(num)
                integer_values.append(num_int)
            except ValueError:
                pass  # Skip non-numeric values
        return integer_values

    # Load the dataset into a Pandas DataFrame
    data = dataset

    dataset = pd.DataFrame(data)

    # Split the column into separate integer columns
    dataset[['Corner height', 'Corner width']] = dataset['Corner column(o/p)'].apply(convert_to_int).apply(pd.Series)

    # Drop the original column if needed
    dataset.drop('Corner column(o/p)', axis=1, inplace=True)

  
    def convert_to_int(value):
        numbers = value.split('X')  # Use 'X' as the separator (capital 'X')
        integer_values = []
        for num in numbers:
            try:
                num_int = int(num)
                integer_values.append(num_int)
            except ValueError:
                pass  # Skip non-numeric values
        return integer_values

    # Load the dataset into a Pandas DataFrame
    data = dataset

    dataset = pd.DataFrame(data)

    # Split the column into separate integer columns
    dataset[['Inner height', 'Inner width']] = dataset['Inner Column(o/p)'].apply(convert_to_int).apply(pd.Series)

    # Drop the original column if needed
    dataset.drop('Inner Column(o/p)', axis=1, inplace=True)



    def convert_to_int(value):
        numbers = value.split('X')  # Use 'X' as the separator (capital 'X')
        integer_values = []
        for num in numbers:
            try:
                num_int = int(num)
                integer_values.append(num_int)
            except ValueError:
                pass  # Skip non-numeric values
        return integer_values

    # Load the dataset into a Pandas DataFrame
    data = dataset

    dataset = pd.DataFrame(data)

    # Split the column into separate integer columns
    dataset[['Beam height', 'Beam width']] = dataset['Beam(o/p)'].apply(convert_to_int).apply(pd.Series)

    # Drop the original column if needed
    dataset.drop('Beam(o/p)', axis=1, inplace=True)

    dataset.dropna(inplace=True)

# Load the dataset
    data = dataset
    

    df = pd.DataFrame(data)

    # Split the dataset into input features and target variables
    X = df[['Span(i/p)', 'Storey(i/p)', 'Area(sqm)(i/p)', 'Outer height', 'Outer width']]
    y_corner_height = df['Corner height']
    y_corner_width = df['Corner width']
    y_inner_height = df['Inner height']
    y_inner_width = df['Inner width']
    y_outer_height = df['Outer height']
    y_outer_width = df['Outer width']
    y_beam_height = df['Beam height']
    y_beam_width = df['Beam width']

    # Create and train the SVR models for each dimension
    svr_corner_height = SVR()
    svr_corner_height.fit(X, y_corner_height)

    svr_corner_width = SVR()
    svr_corner_width.fit(X, y_corner_width)

    svr_inner_height = SVR()
    svr_inner_height.fit(X, y_inner_height)

    svr_inner_width = SVR()
    svr_inner_width.fit(X, y_inner_width)

    svr_outer_height = SVR()
    svr_outer_height.fit(X, y_outer_height)

    svr_outer_width = SVR()
    svr_outer_width.fit(X, y_outer_width)

    svr_beam_height = SVR()
    svr_beam_height.fit(X, y_beam_height)

    svr_beam_width = SVR()
    svr_beam_width.fit(X, y_beam_width)

    # Create a dataframe with the input values for prediction
    input_data = pd.DataFrame({
        'Span(i/p)': [span],
        'Storey(i/p)': [storey],
        'Area(sqm)(i/p)': [area],
        'Outer height': [0.0],
        'Outer width': [0.0]
    })
    print(input_data)
    # Predict the dimensions
    corner_height_prediction = svr_corner_height.predict(input_data)
    corner_width_prediction = svr_corner_width.predict(input_data)
    inner_height_prediction = svr_inner_height.predict(input_data)
    inner_width_prediction = svr_inner_width.predict(input_data)
    outer_height_prediction = svr_outer_height.predict(input_data)
    outer_width_prediction = svr_outer_width.predict(input_data)
    beam_height_prediction = svr_beam_height.predict(input_data)
    beam_width_prediction = svr_beam_width.predict(input_data)

    # Print the predicted dimensions
    print("L shape:- ")
    print("Inner Dimension: {}x{}".format(round(inner_height_prediction[0]), round(inner_width_prediction[0])))
    print("Outer Dimension: {}x{}".format(round(outer_height_prediction[0]), round(outer_width_prediction[0])))
    print("Corner Dimension: {}x{}".format(round(corner_height_prediction[0]), round(corner_width_prediction[0])))
    print("Beam Dimension: {}x{}".format(round(beam_height_prediction[0]), round(beam_width_prediction[0])))







    
    value={
         'inner_dim':f"{round(inner_height_prediction[0])}x{round(inner_width_prediction[0])}",
          'outer_dim':f"{round(outer_height_prediction[0])}x{round(outer_width_prediction[0])}",
          'corner_dim':f"{round(corner_height_prediction[0])}x{round(corner_width_prediction[0])}",
          'beam_dim':f"{round(beam_height_prediction[0])}x{round(beam_width_prediction[0])}",
    }

    return value


def Lresult(request):
     if request.method == 'GET':
        area = int(request.GET.get('area', 0))
        shape = int(request.GET.get('span', 0))
        storey = int(request.GET.get('storey', 0))
        # res = prediction(area, shape, noofguids)
        
        Lres=Lshapeprediction(area,shape,storey)
      
        context = { 
                    
                   'Lres':Lres
                   }
     
        return render(request, 'Lresult.html', context)
     




def Cshape(request):
    return render(request,'Cshape.html')

def Cshapeprediction(area,shape,storey):
    dataset = pd.read_csv(r"C:/Users/Atees/Desktop/djangoprojects/ml/construction/static/C.csv")

   
    def convert_to_int(value):
        numbers = value.split('x')  # Use 'x' as the separator
        integer_values = []
        for num in numbers:
            try:
                num_int = int(num)
                integer_values.append(num_int)
            except ValueError:
                pass  # Skip non-numeric values
        return integer_values

    # Load the dataset into a Pandas DataFrame


    # Split the column into separate integer columns
    dataset[['Outer height', 'Outer width']] = dataset['Outer column(o/p)'].apply(convert_to_int).apply(pd.Series)

    # Drop the original column if needed
    dataset.drop('Outer column(o/p)', axis=1, inplace=True)




    def convert_to_int(value):
        numbers = value.split('x')  # Use 'x' as the separator
        integer_values = []
        for num in numbers:
            try:
                num_int = int(num)
                integer_values.append(num_int)
            except ValueError:
                pass  # Skip non-numeric values
        return integer_values

    # Load the dataset into a Pandas DataFrame


    # Split the column into separate integer columns
    dataset[['Corner height', 'Corner width']] = dataset['Corner column(o/p)'].apply(convert_to_int).apply(pd.Series)

    # Drop the original column if needed
    dataset.drop('Corner column(o/p)', axis=1, inplace=True)


  
   
    def convert_to_int(value):
        numbers = value.split('x')  # Use 'x' as the separator
        integer_values = []
        for num in numbers:
            try:
                num_int = int(num)
                integer_values.append(num_int)
            except ValueError:
                pass  # Skip non-numeric values
        return integer_values


    # Split the column into separate integer columns
    dataset[['Inner height', 'Inner width']] = dataset['Inner Column(o/p)'].apply(convert_to_int).apply(pd.Series)

    # Drop the original column if needed
    dataset.drop('Inner Column(o/p)', axis=1, inplace=True) 



    def convert_to_int(value):
        numbers = value.split('x')  # Use 'x' as the separator
        integer_values = []
        for num in numbers:
            try:
                num_int = int(num)
                integer_values.append(num_int)
            except ValueError:
                pass  # Skip non-numeric values
        return integer_values


    # Split the column into separate integer columns
    dataset[['Beam height', 'Beam width']] = dataset['Beam(o/p)'].apply(convert_to_int).apply(pd.Series)

    # Drop the original column if needed
    dataset.drop('Beam(o/p)', axis=1, inplace=True)



    dataset.fillna(0, inplace=True)

    # Split the dataset into input features and target variables
    X = dataset[['Span(i/p)', 'Storey(i/p)', 'Area(sqm)(i/p)']]
    y = dataset[['Inner height', 'Inner width', 'Outer height', 'Outer width', 'Corner height', 'Corner width', 'Beam height', 'Beam width']]

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate the mean squared error (MSE)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    # Predict dimensions for new data
    new_data = {
        'Span(i/p)': [shape],
        'Storey(i/p)': [storey],
        'Area(sqm)(i/p)': [area],
    }
    new_df = pd.DataFrame(new_data)
    print(new_df)

    predicted_dimensions = model.predict(new_df)
    inner_dim = predicted_dimensions[0][:2].astype(int)
    outer_dim = predicted_dimensions[0][2:4].astype(int)
    corner_dim = predicted_dimensions[0][4:6].astype(int)
    beam_dim = predicted_dimensions[0][6:].astype(int)

    print("C shape:-")
    print(f"Inner Dimension: {max(inner_dim[0], 250)}x{max(inner_dim[1], 250)}")
    print(f"Outer Dimension: {max(outer_dim[0], 250)}x{max(outer_dim[1], 250)}")
    print(f"Corner Dimension: {max(corner_dim[0], 250)}x{max(corner_dim[1], 250)}")
    print(f"Beam Dimension: {max(beam_dim[0], 250)}x{max(beam_dim[1], 250)}")
        
    value={
         'inner_dim':f"{max(inner_dim[0], 250)}x{max(inner_dim[1], 250)}",
          'outer_dim':f"{max(outer_dim[0],250)}x{max(outer_dim[1], 250)}",
          'corner_dim':f"{max(corner_dim[0], 250)}x{max(corner_dim[1], 250)}",
          'beam_dim':f"{max(beam_dim[0], 250)}x{max(beam_dim[1], 250)}",
    }
    return value
    print(value)

def Cresult(request):
     if request.method == 'GET':
        area = int(request.GET.get('area', 0))
        shape = int(request.GET.get('shape', 0))
        noofguids = int(request.GET.get('storey', 0))
        # res = prediction(area, shape, noofguids)
        
        Cres=Cshapeprediction(area,shape,shape)
      
        context = { 
                    
                   'Cres':Cres
                   }
     
        return render(request, 'Cresult.html', context)


def Tshape(request):
    return render(request,'Tshape.html')

def Tshapeprediction(area,span,storey):
    dataset = pd.read_csv(r"C:/Users/Atees/Desktop/djangoprojects/ml/construction/static/T.csv")
    def convert_to_int(value):
        numbers = value.split('X')  # Use 'X' as the separator (capital 'X')
        integer_values = []
        for num in numbers:
            try:
                num_int = int(num)
                integer_values.append(num_int)
            except ValueError:
                pass  # Skip non-numeric values
        return integer_values

    # Load the dataset into a Pandas DataFrame
    data = dataset

    dataset = pd.DataFrame(data)

    # Split the column into separate integer columns
    dataset[['Outer height', 'Outer width']] = dataset['Outer column(o/p)'].apply(convert_to_int).apply(pd.Series)

    # Drop the original column if needed
    dataset.drop('Outer column(o/p)', axis=1, inplace=True)



    def convert_to_int(value):
            numbers = value.split('X')  # Use 'X' as the separator (capital 'X')
            integer_values = []
            for num in numbers:
                try:
                    num_int = int(num)
                    integer_values.append(num_int)
                except ValueError:
                    pass  # Skip non-numeric values
            return integer_values

        # Load the dataset into a Pandas DataFrame
    data = dataset

    dataset = pd.DataFrame(data)

        # Split the column into separate integer columns
    dataset[['Inner height', 'Inner width']] = dataset['Inner Column(o/p)'].apply(convert_to_int).apply(pd.Series)

        # Drop the original column if needed
    dataset.drop('Inner Column(o/p)', axis=1, inplace=True)

    print(dataset)




    def convert_to_int(value):
        numbers = value.split('X')  # Use 'X' as the separator (capital 'X')
        integer_values = []
        for num in numbers:
            try:
                num_int = int(num)
                integer_values.append(num_int)
            except ValueError:
                pass  # Skip non-numeric values
        return integer_values

    # Load the dataset into a Pandas DataFrame
    data = dataset

    dataset = pd.DataFrame(data)

    # Split the column into separate integer columns
    dataset[['Corner height', 'Corner width']] = dataset['Corner column(o/p)'].apply(convert_to_int).apply(pd.Series)

    # Drop the original column if needed
    dataset.drop('Corner column(o/p)', axis=1, inplace=True)


    def convert_to_int(value):
            numbers = value.split('X')  # Use 'X' as the separator (capital 'X')
            integer_values = []
            for num in numbers:
                try:
                    num_int = int(num)
                    integer_values.append(num_int)
                except ValueError:
                    pass  # Skip non-numeric values
            return integer_values

        # Load the dataset into a Pandas DataFrame
    data = dataset

    dataset = pd.DataFrame(data)

        # Split the column into separate integer columns
    dataset[['Beam height', 'Beam width']] = dataset['Beam(o/p)'].apply(convert_to_int).apply(pd.Series)

        # Drop the original column if needed
    dataset.drop('Beam(o/p)', axis=1, inplace=True)


    dataset.dropna(inplace=True)

        # Load the dataset
        # Split the dataset into input features and target variables
    X = dataset[['Span(i/p)', 'Storey(i/p)', 'Area(sqm)(i/p)']]
    y = dataset[['Inner height', 'Inner width', 'Outer height', 'Outer width', 'Corner height', 'Corner width', 'Beam height', 'Beam width']]

    # Train the linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Predict dimensions for new data
    new_data = {
        'Span(i/p)': [shape],
        'Storey(i/p)': [storey],
        'Area(sqm)(i/p)': [area],
    }
    new_df = pd.DataFrame(new_data)
    print(new_data)

    predicted_dimensions = model.predict(new_df)
    inner_dim = predicted_dimensions[0][:2].astype(int)
    outer_dim = predicted_dimensions[0][2:4].astype(int)
    corner_dim = predicted_dimensions[0][4:6].astype(int)
    beam_dim = predicted_dimensions[0][6:].astype(int)
    print(beam_dim)

    print("T shape:-")
    print(f"Inner Dimension: {max(inner_dim[0], 250)}x{max(inner_dim[1], 250)}")
    print(f"Outer Dimension: {max(outer_dim[0], 250)}x{max(outer_dim[1], 250)}")
    print(f"Corner Dimension: {max(corner_dim[0], 250)}x{max(corner_dim[1], 250)}")
    print(f"Beam Dimension: {max(beam_dim[0], 250)}x{max(beam_dim[1], 250)}")
        
    value={
         'inner_dim':f" {max(inner_dim[0], 250)}x{max(inner_dim[1], 300)}",
          'outer_dim':f"{max(outer_dim[0], 250)}x{max(outer_dim[1], 300)}",
          'corner_dim':f"{max(corner_dim[0], 250)}x{max(corner_dim[1], 300)}",
          'beam_dim':f"{max(beam_dim[0], 250)}x{max(beam_dim[1], 250)}",
    }
    return value


def Tresult(request):
     if request.method == 'GET':
        area = int(request.GET.get('area', 0))
        shape = int(request.GET.get('span', 0))
        storey = int(request.GET.get('storey', 0))
        # res = prediction(area, shape, noofguids)
        
        Tres=Tshapeprediction(area,shape,storey)
      
        context = { 
                    
                   'Tres':Tres
                   }
     
        return render(request, 'Tresult.html', context)





def Ishape(request):
    return render(request,'Ishape.html')


def Ishapeprediction(area,shape,storey):
    dataset = pd.read_csv(r"C:/Users/Atees/Desktop/djangoprojects/ml/construction/static/I.csv")
    def convert_to_int(value):
        numbers = value.split('X')  # Use 'X' as the separator (capital 'X')
        integer_values = []
        for num in numbers:
            try:
                num_int = int(num)
                integer_values.append(num_int)
            except ValueError:
                pass  # Skip non-numeric values
        return integer_values

    # Load the dataset into a Pandas DataFrame
    data = dataset

    dataset = pd.DataFrame(data)

    # Split the column into separate integer columns
    dataset[['Outer height', 'Outer width']] = dataset['Outer column(o/p)'].apply(convert_to_int).apply(pd.Series)

    # Drop the original column if needed
    dataset.drop('Outer column(o/p)', axis=1, inplace=True)







    def convert_to_int(value):
        numbers = value.split('X')  # Use 'X' as the separator (capital 'X')
        integer_values = []
        for num in numbers:
            try:
                num_int = int(num)
                integer_values.append(num_int)
            except ValueError:
                pass  # Skip non-numeric values
        return integer_values

    # Load the dataset into a Pandas DataFrame
    data = dataset

    dataset = pd.DataFrame(data)

    # Split the column into separate integer columns
    dataset[['Corner height', 'Corner width']] = dataset['Corner column(o/p)'].apply(convert_to_int).apply(pd.Series)

    # Drop the original column if needed
    dataset.drop('Corner column(o/p)', axis=1, inplace=True)

   



    def convert_to_int(value):
        numbers = value.split('X')  # Use 'X' as the separator (capital 'X')
        integer_values = []
        for num in numbers:
            try:
                num_int = int(num)
                integer_values.append(num_int)
            except ValueError:
                pass  # Skip non-numeric values
        return integer_values

    # Load the dataset into a Pandas DataFrame
    data = dataset

    dataset = pd.DataFrame(data)

    # Split the column into separate integer columns
    dataset[['Inner height', 'Inner width']] = dataset['Inner Column(o/p)'].apply(convert_to_int).apply(pd.Series)

    # Drop the original column if needed
    dataset.drop('Inner Column(o/p)', axis=1, inplace=True)

 





    def convert_to_int(value):
        numbers = value.split('X')  # Use 'X' as the separator (capital 'X')
        integer_values = []
        for num in numbers:
            try:
                num_int = int(num)
                integer_values.append(num_int)
            except ValueError:
                pass  # Skip non-numeric values
        return integer_values

    # Load the dataset into a Pandas DataFrame
    data = dataset

    dataset = pd.DataFrame(data)

    # Split the column into separate integer columns
    dataset[['Beam height', 'Beam width']] = dataset['Beam(o/p)'].apply(convert_to_int).apply(pd.Series)

    # Drop the original column if needed
    dataset.drop('Beam(o/p)', axis=1, inplace=True)

   





    dataset.dropna(inplace=True)

    # Split the dataset into input features and target variables
    X = dataset[['Span(i/p)', 'Storey(i/p)', 'Area(sqm)(i/p)']]
    y = dataset[['Inner height', 'Inner width', 'Outer height', 'Outer width', 'Corner height', 'Corner width', 'Beam height', 'Beam width']]

    # Train the linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Predict dimensions for new data
    new_data = {
        'Span(i/p)': [shape],
        'Storey(i/p)': [storey],
        'Area(sqm)(i/p)': [area],
    }
    print(new_data)
    new_df = pd.DataFrame(new_data)

    predicted_dimensions = model.predict(new_df)
    inner_dim = predicted_dimensions[0][:2].astype(int)
    outer_dim = predicted_dimensions[0][2:4].astype(int)
    corner_dim = predicted_dimensions[0][4:6].astype(int)
    beam_dim = predicted_dimensions[0][6:].astype(int)
    print(inner_dim)
    print(outer_dim)
    print(corner_dim)
    print(beam_dim)
    print("I shape:-")
    print(f"Inner Dimension: {max(inner_dim[0], 250)}x{max(inner_dim[1], 250)}")
    print(f"Outer Dimension: {max(outer_dim[0], 250)}x{max(outer_dim[1], 250)}")
    print(f"Corner Dimension: {max(corner_dim[0], 250)}x{max(corner_dim[1], 250)}")
    print(f"Beam Dimension: {max(beam_dim[0], 250)}x{max(beam_dim[1], 250)}")

    print(dataset)
    
    value={
         'inner_dim':f"{max(inner_dim[0], 250)}x{max(inner_dim[1], 250)}",
          'outer_dim':f"{max(outer_dim[0], 250)}x{max(outer_dim[1], 250)}",
          'corner_dim':f"{max(corner_dim[0], 250)}x{max(corner_dim[1], 250)}",
          'beam_dim':f"{max(beam_dim[0], 250)}x{max(beam_dim[1], 250)}",
    }
    return value

def Iresult(request):
     if request.method == 'GET':
        area = int(request.GET.get('area', 0))
        shape = int(request.GET.get('shape', 0))
        noofguids = int(request.GET.get('storey', 0))
        # res = prediction(area, shape, noofguids)
        
        Ires=Ishapeprediction(area,shape,noofguids)
      
        context = { 
                    
                   'Ires':Ires
                   }
     
        return render(request, 'Iresult.html', context)


