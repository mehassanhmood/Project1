class FeatureSelector:
    def __init__(self,x,y,model):
        self.x =x
        self.y=y
        self.model = model

    def select():
        from mlxtend.feature_selection import SequentialFeatureSelector
        bfs = SequencialFeatureSelector(self.model,k_features="best", forward=False , n_jobs=1)
        bfs.fit(self.x,self.y)
        return list(map(int,list(bfs.k_feature_names_)))
