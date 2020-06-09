function FValue_SVM = compute_FValue_ConfusionMat(ConMat) 

    SVM_TP = ConMat(1,1); 
    SVM_TN = ConMat(2,2); 
    SVM_FP = ConMat(2,1); 
    SVM_FN = ConMat(1,2); 

    precision = SVM_TP / (SVM_TP + SVM_FP);
    recall = SVM_TP/ (SVM_TP + SVM_FN);
    FValue_SVM = 2 * precision * recall / (precision + recall);
