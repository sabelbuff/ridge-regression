import numpy as np
import random
import scipy.io as io
import copy


class RidgeReg(object):
    def __init__(self, x_train, y_train, x_test, y_test, degree,  batch_size, learning_rate, lambda_set, epoch):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.batch_size = batch_size
        self.lambda_set = lambda_set
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.degree = degree

    def basis_function_transform_poly(self, x):
        new_x = []
        for sample in x:
            temp_array = [1]
            for feature in sample:
                for d in range(1, self.degree + 1):
                    temp_array.append(feature ** d)
            new_x.append(temp_array)
        return new_x

    def hypothesis(self, x, w):
        return np.matmul(x, w)

    def least_square_error(self, x, y, w, lamb):
        h = self.hypothesis(x, w)
        error = y - h
        ls = np.square(error)
        return np.sum(ls)/x.shape[0] + (lamb*np.sum(np.square(w)))

    def gradient(self, x, y, w, lamb):
        return (np.matmul(np.matmul(x.T, x), w) - np.matmul(x.T, y))*2 + 2*lamb*np.sum(np.abs(w))

    def cf_solution(self):
        x_train_transform = self.basis_function_transform_poly(self.x_train)
        x = np.asmatrix(np.array(x_train_transform))
        # x = np.asmatrix(np.array(self.x_train))
        y = np.asmatrix(np.array(self.y_train))
        x_test = np.asmatrix(np.array(self.basis_function_transform_poly(self.x_test)))
        # x_test = np.asmatrix(np.array(self.x_test))
        y_test = np.asmatrix(np.array(self.y_test))
        if not isinstance(self.lambda_set, list):
            lambda_reg = self.lambda_set
        else:
            lambda_reg = self.lambda_set[0]
        # lambda_reg = 0.0001
        weight = np.matmul(np.matmul(np.asmatrix(np.matmul(x.T, x)).I, x.T), y)
        training_error = self.least_square_error(x, y, weight, lambda_reg)
        print("Training_error = ", training_error)
        testing_error = self.least_square_error(x_test, y_test, weight, lambda_reg)
        print("Testing_error = ", testing_error)
        print(weight)
        # return weight

    def sgd_training(self, x_train, y_train, weight, epoch, learning_rate, lambda_reg):
        train_samples = list(zip(x_train, y_train))
        # num_of_samples = y_train.shape[0]
        i = 1
        while i <= epoch:
            if i % 1000 == 0:
                training_error = self.least_square_error(x_train, y_train, weight, lambda_reg)
                print("Epoch : ", i, "; ", "Training_error = ", training_error)
            random.shuffle(train_samples)
            train_copy = copy.deepcopy(train_samples)
            while len(train_copy) != 0:
                train_batch = train_copy[:self.batch_size]
                train_copy = train_copy[self.batch_size:]
                for (x_temp, y_temp) in train_batch:
                    weight = weight - learning_rate * self.gradient(x_temp, y_temp, weight, lambda_reg)/self.batch_size
            i += 1
        # print(weight)
        return weight

    def sgd(self):
        x_train_transform = self.basis_function_transform_poly(self.x_train)
        x = np.asmatrix(np.array(x_train_transform))
        x_test = np.asmatrix(np.array(self.basis_function_transform_poly(self.x_test)))
        y = np.asmatrix(np.array(self.y_train))
        y_test = np.asmatrix(np.array(self.y_test))
        num_of_features = x.shape[1]
        if not isinstance(self.lambda_set, list):
            lambda_reg = self.lambda_set
        else:
            lambda_reg = self.lambda_set[0]
        # lambda_reg = 0.003
        weight = np.asmatrix(np.random.randn(num_of_features)).T
        weight_new = self.sgd_training(x, y, weight, self.epoch, self.learning_rate, lambda_reg)
        training_error = self.least_square_error(x, y, weight_new, lambda_reg)
        print("Training_error = ", training_error)
        testing_error = self.least_square_error(x_test, y_test, weight_new, lambda_reg)
        print("Testing_error = ", testing_error)
        print(weight_new)
        # return weight_new

    def sgd_using_cross_validation(self, k):
        x_train_transform = self.basis_function_transform_poly(self.x_train)
        x_test_transform = self.basis_function_transform_poly(self.x_test)
        x = np.asmatrix(np.array(x_train_transform))
        x_test = np.asmatrix(np.array(x_test_transform))
        # x = np.asmatrix(np.array(x_train_transform))
        y = np.asmatrix(np.array(self.y_train))
        # x_test = np.asmatrix(np.array(x_test_transform))
        y_test = np.asmatrix(np.array(self.y_test))
        train_samples = list(zip(x, y))
        num_in_one_fold = int(len(train_samples)/k)
        # total_val_error = 0
        min_val_error = 10000000000
        num_of_features = x.shape[1]
        lambda_reg = self.lambda_set[0]
        for l in self.lambda_set:
            # print(int(num_in_one_fold))
            print("lambda = ", l)
            total_val_error = 0
            for i in range(0, len(train_samples), num_in_one_fold):
                # i += num_in_one_fold
                j = i + num_in_one_fold
                # print(i, j)
                # print(train_samples[:i])
                # print(train_samples[j:])
                train_batch = train_samples[:i] + train_samples[j:]
                validation_batch = train_samples[i:j]
                x_temp = np.asmatrix(np.array(list(zip(*train_batch))[0]))
                # print(y)
                y_temp = np.asmatrix(np.array(list(zip(*train_batch))[1])).T
                x_val = np.asmatrix(np.array(list(zip(*validation_batch))[0]))
                # print(x_val.shape)

                y_val = np.asmatrix(np.array(list(zip(*validation_batch))[1])).T
                # print(y_val.shape)
                weight = np.asmatrix(np.random.randn(num_of_features)).T
                weight_new = self.sgd_training(x_temp, y_temp, weight, self.epoch, self.learning_rate, l)
                # print(weight_new.shape)
                validation_error = self.least_square_error(x_val, y_val, weight_new, l)
                # print(validation_error)
                total_val_error += validation_error
            avg_val_error = total_val_error/k

            print("Validation_error = ", avg_val_error)
            if avg_val_error < min_val_error:
                min_val_error = avg_val_error
                lambda_reg = l
        print("training.......")
        print(lambda_reg)
        weight = np.asmatrix(np.random.randn(num_of_features)).T
        weight_new = self.sgd_training(x, y, weight, self.epoch, self.learning_rate, lambda_reg)
        training_error = self.least_square_error(x, y, weight_new, lambda_reg)
        print("Training_error = ", training_error)
        testing_error = self.least_square_error(x_test, y_test, weight_new, lambda_reg)
        print("Testing_error = ", testing_error)
        print(weight_new)
        # return weight_new


data = io.loadmat("dataset2.mat")
X_train = data['X_trn']
Y_train = data['Y_trn']
X_test = data['X_tst']
Y_test = data['Y_tst']

lambda_set = [0.00001, 0.0001, 0.1, 0.003, 0.001, 0.03]
test = RidgeReg(X_train, Y_train, X_test, Y_test, 2, 10, 0.00000001, lambda_set, 10000)
test.cf_solution()
test.sgd()
test.sgd_using_cross_validation()




