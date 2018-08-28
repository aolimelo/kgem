from argparse import ArgumentParser
import time
from tools import PairwiseEval, MultilabelEval, load_data, ranking_scores
from models import TransE_PairwiseMargin, TransR_PairwiseMargin, ConvE_PairwiseMargin, HolE_PairwiseMargin, DistMult_PairwiseMargin, \
    ProjE_PairwiseMargin, TransE_OptimizeRank, ConvE_OptimizeRank, TransE_Multilabel, ComplexE_PairwiseMargin, DistMult_OptimizeRank, \
    ComplexE_OptimizeRank, HolE_OptimizeRank


def eval(train_triples, test_triples, valid_triples, entity2id, relation2id, n_dim=100, n_neg=15, layers=None,
         max_pool_dim=2, fitgen=False, reg=0.0, constraint=0,
         sample_mode="corrupted", verbose=False, n_epochs=None, methods=["conve"], training_methods=["pw"],
         filter_dim=3, stride=1, comb="diff"):
    all_triples = train_triples + test_triples + valid_triples

    print(n_neg)
    for method in methods:
        for tm in training_methods:

            # Pairwise Margin
            if tm == "pw":
                if method == "conve":
                    ed = ConvE_PairwiseMargin(n_filters=layers[0], n_dim=n_dim, n_instances=len(entity2id),
                                              n_relations=len(relation2id),
                                              n_neg=n_neg,
                                              verbose=verbose, layers=layers, epochs=n_epochs, sample_mode=sample_mode,
                                              regularizer=reg, constraint=constraint)
                if method == "transe":
                    ed = TransE_PairwiseMargin(n_dim=n_dim, n_instances=len(entity2id), n_relations=len(relation2id),
                                               n_neg=n_neg,
                                               verbose=verbose, layers=layers, epochs=n_epochs, sample_mode=sample_mode,
                                               regularizer=reg, constraint=constraint)

                if method == "transr":
                    ed = TransR_PairwiseMargin(n_rel_dim=layers[0], n_dim=n_dim, n_instances=len(entity2id), n_relations=len(relation2id),
                                               n_neg=n_neg,
                                               verbose=verbose, layers=layers, epochs=n_epochs, sample_mode=sample_mode,
                                               regularizer=reg, constraint=constraint)

                if method == "hole":
                    ed = HolE_PairwiseMargin(n_dim=n_dim, n_instances=len(entity2id), n_relations=len(relation2id),
                                             n_neg=n_neg,
                                             verbose=verbose, layers=layers, epochs=n_epochs, sample_mode=sample_mode,
                                             regularizer=reg, constraint=constraint)

                if method == "distmult":
                    ed = DistMult_PairwiseMargin(n_dim=n_dim, n_instances=len(entity2id), n_relations=len(relation2id),
                                                 n_neg=n_neg,
                                                 verbose=verbose, layers=layers, epochs=n_epochs,
                                                 sample_mode=sample_mode,
                                                 regularizer=reg, constraint=constraint)

                if method == "complexe":
                    ed = ComplexE_PairwiseMargin(n_dim=n_dim, n_instances=len(entity2id), n_relations=len(relation2id),
                                                 n_neg=n_neg,
                                                 verbose=verbose, layers=layers, epochs=n_epochs,
                                                 sample_mode=sample_mode,
                                                 regularizer=reg, constraint=constraint)

                if method == "proje":
                    ed = ProjE_PairwiseMargin(n_dim=n_dim, n_instances=len(entity2id), n_relations=len(relation2id),
                                              n_neg=n_neg,
                                              verbose=verbose, layers=layers, epochs=n_epochs, sample_mode=sample_mode,
                                              regularizer=reg, constraint=constraint)

            # Optimization Rank
            if tm == "or":
                if method == "transe":
                    ed = TransE_OptimizeRank(n_dim=n_dim, n_instances=len(entity2id), n_relations=len(relation2id),
                                             n_neg=n_neg,
                                             verbose=verbose, layers=layers, epochs=n_epochs, sample_mode=sample_mode,
                                             regularizer=reg, constraint=constraint)

                if method == "distmult":
                    ed = DistMult_OptimizeRank(n_dim=n_dim, n_instances=len(entity2id), n_relations=len(relation2id),
                                             n_neg=n_neg,
                                             verbose=verbose, layers=layers, epochs=n_epochs, sample_mode=sample_mode,
                                             regularizer=reg, constraint=constraint)

                if method == "hole":
                    ed = HolE_OptimizeRank(n_dim=n_dim, n_instances=len(entity2id), n_relations=len(relation2id),
                                           n_neg=n_neg,
                                           verbose=verbose, layers=layers, epochs=n_epochs, sample_mode=sample_mode,
                                           regularizer=reg, constraint=constraint)

                if method == "conve":
                    ed = ConvE_OptimizeRank(n_filters=layers[0], n_dim=n_dim, n_instances=len(entity2id),
                                            n_relations=len(relation2id),
                                            n_neg=n_neg,
                                            verbose=verbose, layers=layers, epochs=n_epochs, sample_mode=sample_mode,
                                            regularizer=reg, constraint=constraint)

                if method == "complexe":
                    ed = ComplexE_OptimizeRank(n_dim=n_dim, n_instances=len(entity2id), n_relations=len(relation2id),
                                                 n_neg=n_neg,
                                                 verbose=verbose, layers=layers, epochs=n_epochs,
                                                 sample_mode=sample_mode,
                                                 regularizer=reg, constraint=constraint)

            # Multilabel Classification
            if tm == "ml":
                if method == "transe":
                    ed = TransE_Multilabel(n_dim=n_dim, n_instances=len(entity2id),
                                             n_relations=len(relation2id),
                                             n_neg=n_neg,
                                             verbose=verbose, layers=layers, epochs=n_epochs,
                                             sample_mode=sample_mode,
                                             regularizer=reg, constraint=constraint)

        start = time.time()
        if fitgen:
            ed.fit_generator(train_triples, valid_triples)
        else:
            ed.fit(train_triples, valid_triples)
        end = time.time()
        print("learning time = %d ms" % (end - start))

        if tm in ["pw","or"]:
            eval = PairwiseEval(test_triples, all_triples, verbose=verbose)
        if tm == "ml":
            eval = MultilabelEval(test_triples, all_triples, verbose=verbose)
        pos, fpos = eval.positions(ed)
        ranking_scores(pos, fpos)
        ed.close()



if __name__ == '__main__':
    parser = ArgumentParser(description="ConvE: Convolutional Knowledge Graph Embeddings")
    parser.add_argument("input", type=str, default=None, help="path to dataset on which to perform the evaluation")
    parser.add_argument("-m", "--methods", type=str, nargs="+", default=["conve"], help="list of lp methods")
    parser.add_argument("-d", "--dims", type=int, default=100, help="number of dimensions")
    parser.add_argument("-tm", "--train-methods", type=str, nargs="+", default=["pw"],
                        help="embeddings training methods (pw=pairwise, or=optimize_rank, ml=relation_multilabel_classification)")
    parser.add_argument("-ne", "--n-epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("-nf", "--n-filters", type=int, nargs="+", default=None, help="number of filters")
    parser.add_argument("-fd", "--filter-dim", type=int, default=3, help="filter dimension (square [n,n])")
    parser.add_argument("-st", "--stride", type=int, default=1, help="filter stride")
    parser.add_argument("-t", "--triple-order", type=str, default="sop", help="Triple order (sop, spo, pos)")
    parser.add_argument("-mp", "--maxpool-dim", type=int, default=2, help="filter stride")
    parser.add_argument("-nneg", "--n-negatives", type=int, default=15, help="nunber of negative examples per positive")
    parser.add_argument("-sm", "--sample-mode", type=str, default="corrupted", help="sample mode")
    parser.add_argument("-reg", "--regularization", type=float, default=0.0, help="regularization parameter")
    parser.add_argument("-ec", "--embedding-constraint", type=float, default=1.0, help="embedding norm constraint")
    parser.add_argument("-v", "--verbose", dest="verbose", action="store_true", help="verbose mode")
    parser.add_argument("-fg", "--fit-generator", dest="fitgen", action="store_true",
                        help="whether to use fit_generator")

    parser.set_defaults(verbose=False)
    parser.set_defaults(fitgen=False)

    args = parser.parse_args()

    print(args)

    train_triples, test_triples, valid_triples, entity2id, relation2id = load_data(args.input, order=args.triple_order)

    eval(train_triples, test_triples, valid_triples, entity2id, relation2id, n_dim=args.dims,
         sample_mode=args.sample_mode, n_neg=args.n_negatives, verbose=args.verbose, n_epochs=args.n_epochs,
         methods=args.methods, training_methods=args.train_methods, filter_dim=args.filter_dim, stride=args.stride,
         max_pool_dim=args.maxpool_dim,
         layers=args.n_filters, fitgen=args.fitgen, reg=args.regularization,
         constraint=args.embedding_constraint)
