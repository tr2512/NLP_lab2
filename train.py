import torch
import torch.nn as nn

import time
import random
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt

from data import examplereader, build_vocab, filereader, Vocabulary, get_minibatch, prepare_minibatch, prepare_treelstm_minibatch
from model import BOW, CBOW, DeepCBOW, LSTMClassifier, TreeLSTMClassifier

def evaluate(model, data,
             batch_fn=get_minibatch, prep_fn=prepare_minibatch,
             batch_size=16):
  """Accuracy of a model on given data set (using mini-batches)"""
  correct = 0
  total = 0
  model.eval()  # disable dropout

  for mb in batch_fn(data, batch_size=batch_size, shuffle=False):
    x, targets = prep_fn(mb, model.vocab, device)
    with torch.no_grad():
      logits = model(x)

    predictions = logits.argmax(dim=-1).view(-1)

    # add the number of correct predictions to the total correct
    correct += (predictions == targets.view(-1)).sum().item()
    total += targets.size(0)

  return correct, total, correct / float(total)

def train_model(model, optimizer, train_data, dev_data, test_data, num_iterations=10000,
                print_every=1000, eval_every=1000,
                batch_fn=get_minibatch,
                prep_fn=prepare_minibatch,
                eval_fn=evaluate,
                batch_size=1, eval_batch_size=None,
                file_name_suffix=""):
  """Train a model."""
  iter_i = 0
  train_loss = 0.
  print_num = 0
  start = time.time()
  criterion = nn.CrossEntropyLoss() # loss function
  best_eval = 0.
  best_iter = 0

  # store train loss and validation accuracy during training
  # so we can plot them afterwards
  losses = []
  accuracies = []

  if eval_batch_size is None:
    eval_batch_size = batch_size

  while True:  # when we run out of examples, shuffle and continue
    for batch in batch_fn(train_data, batch_size=batch_size):

      # forward pass
      model.train()
      x, targets = prep_fn(batch, model.vocab, device)
      logits = model(x)

      B = targets.size(0)  # later we will use B examples per update

      # compute cross-entropy loss (our criterion)
      # note that the cross entropy loss function computes the softmax for us
      loss = criterion(logits.view([B, -1]), targets.view(-1))
      train_loss += loss.item()

      # backward pass (tip: check the Introduction to PyTorch notebook)

      # erase previous gradients
      #raise NotImplementedError("Implement this")
      # YOUR CODE HERE
      optimizer.zero_grad()
      # compute gradients
      # YOUR CODE HERE
      loss.backward()
      # update weights - take a small step in the opposite dir of the gradient
      # YOUR CODE HERE
      optimizer.step()
      print_num += 1
      iter_i += 1

      # print info
      if iter_i % print_every == 0:
        print("Iter %r: loss=%.4f, time=%.2fs" %
              (iter_i, train_loss, time.time()-start))
        losses.append(train_loss)
        print_num = 0
        train_loss = 0.

      # evaluate
      if iter_i % eval_every == 0:
        _, _, accuracy = eval_fn(model, dev_data, batch_size=eval_batch_size,
                                 batch_fn=batch_fn, prep_fn=prep_fn)
        accuracies.append(accuracy)
        print("iter %r: dev acc=%.4f" % (iter_i, accuracy))

        # save best model parameters
        if accuracy > best_eval:
          print("new highscore")
          best_eval = accuracy
          best_iter = iter_i
          path = "{}{}.pt".format(model.__class__.__name__, file_name_suffix)
          ckpt = {
              "state_dict": model.state_dict(),
              "optimizer_state_dict": optimizer.state_dict(),
              "best_eval": best_eval,
              "best_iter": best_iter
          }
          torch.save(ckpt, path)

      # done training
      if iter_i == num_iterations:
        print("Done training")

        # evaluate on train, dev, and test with best model
        print("Loading best model")
        path = "{}{}.pt".format(model.__class__.__name__, file_name_suffix)
        ckpt = torch.load(path)
        model.load_state_dict(ckpt["state_dict"])

        _, _, train_acc = eval_fn(
            model, train_data, batch_size=eval_batch_size,
            batch_fn=batch_fn, prep_fn=prep_fn)
        _, _, dev_acc = eval_fn(
            model, dev_data, batch_size=eval_batch_size,
            batch_fn=batch_fn, prep_fn=prep_fn)
        _, _, test_acc = eval_fn(
            model, test_data, batch_size=eval_batch_size,
            batch_fn=batch_fn, prep_fn=prep_fn)

        print("best model iter {:d}: "
              "train acc={:.4f}, dev acc={:.4f}, test acc={:.4f}".format(
                  best_iter, train_acc, dev_acc, test_acc))

        ckpt['losses'] = losses
        ckpt['accuracies'] = accuracies
        path = "{}{}.pt".format(model.__class__.__name__, file_name_suffix)
        torch.save(ckpt, path)

        return losses, accuracies

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Model hyperparameters
    parser.add_argument('--seed', default=0, type=int,
                        help='Model seed')
    parser.add_argument('--model', default='BOW', type=str,
                        help='Supported model: BOW, CBOW, DeepCBOW, LSTM, TreeLSTM')
    parser.add_argument('--embedding', default='Random', type=str,
                        help='Supported pretrained: Random, Word2vec, Glove')
    parser.add_argument('--finetune', action='store_true',
                        help='Finetune the pretrained. Only called if Glove or Word2vec embedding is used')
    parser.add_argument('--num_iterations', default=30000, type=int,
                        help='Number of iterations model trained')
    parser.add_argument('--eval_iterations', default=1000, type=int,
                        help='Eval iterations frequency')
    # Optimizer hyperparameters
    parser.add_argument('--optimizer', default='Adam', type=str,
                        help='Optimizer of the model. Only SGD, Adam and AdamW are supported')
    parser.add_argument('--lr', default=0.0005, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Minibatch size')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='Momentum for optimizer. Only used in SGD')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='Weight decay for optimizer. Only used in SGD and AdamW')
    
    parser.add_argument('--visualise', action='store_true',
                        default='Plot validation accuracies and training loss')
    

    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():  
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False
    # Let's load the data into memory.
    LOWER = False  # we will keep the original casing
    train_data = list(examplereader("trees/train.txt", lower=LOWER))
    dev_data = list(examplereader("trees/dev.txt", lower=LOWER))
    test_data = list(examplereader("trees/test.txt", lower=LOWER))

    print("train", len(train_data))
    print("dev", len(dev_data))
    print("test", len(test_data))

    i2t = ["very negative", "negative", "neutral", "positive", "very positive"]
    t2i = OrderedDict({p : i for p, i in zip(i2t, range(len(i2t)))})

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.embedding == 'Random':
        v = build_vocab(train_data)
    elif args.embedding == 'Glove' or args.embedding == 'Word2vec':
        vectors = []
        v = Vocabulary()
        filename = './glove.840B.300d.sst.txt' if args.embedding == 'Glove' else './googlenews.word2vec.300d.txt'
        for word_embedding_pair in filereader(filename):
            entries = word_embedding_pair.split(' ')
            word = entries[0]
            embedding = [float(feature) for feature in entries[1:]]
            vectors.append(embedding)
            v.count_token(word)
        embedding_dim = len(vectors[0])

        unk_embedding = np.zeros((1, embedding_dim))
        pad_embedding = np.ones((1, embedding_dim))

        vectors = np.stack(vectors, axis=0)
        vectors = np.vstack([unk_embedding, pad_embedding, vectors])

        v.build()
        print("Vocabulary size:", len(v.w2i))

        word2vec_vectors = np.array(vectors)
        print("Embedding matrix shape:", word2vec_vectors.shape)
    else:
       raise ValueError(f'{args.embedding} embedding is not supported')

    if args.model == 'BOW':
        model = BOW(len(v.w2i), len(t2i), vocab=v)
    elif args.model == 'CBOW':
        model = CBOW(len(v.w2i), 300, 5, vocab=v)
    elif args.model == 'DeepCBOW':
        model = DeepCBOW(vocab_size = len(v.w2i), embedding_dim = 300, output_size = 5, vocab=v, hidden_dim = 100)
    elif args.model == 'LSTM':
        model = LSTMClassifier(len(v.w2i), 300, 168, len(t2i), v)
    elif args.model == 'TreeLSTM':
        model = TreeLSTMClassifier(len(v.w2i), 300, 150, len(t2i), v)
       
    print(model)
    model.to(device)
    if args.embedding == 'Glove' or args.embedding == 'Word2vec':
        model.embed.weight.data.copy_(torch.from_numpy(vectors))
        if args.finetune:
            model.embed.weight.requires_grad = True
        else:
            model.embed.weight.requires_grad = False
    
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError(f'{args.optimizer} optimizer is not supported')

    if args.model == 'TreeLSTM':
       losses, accuracies = train_model(
        model, optimizer, train_data, dev_data, test_data, num_iterations=args.num_iterations,
        print_every=args.eval_iterations, eval_every=args.eval_iterations, batch_size=args.batch_size, prep_fn=prepare_treelstm_minibatch) 
    else:
        losses, accuracies = train_model(
            model, optimizer, train_data, dev_data, test_data, num_iterations=args.num_iterations,
            print_every=args.eval_iterations, eval_every=args.eval_iterations, batch_size=args.batch_size) 
    if args.visualise:
        fig, ax1 = plt.subplots()

        # Plot training loss (on the left y-axis)
        ax1.plot(losses, color='tab:blue', label='Training Loss')
        ax1.set_xlabel(f'Iterations x {args.eval_iterations}')
        ax1.set_ylabel('Training Loss', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        # Create a second y-axis to plot validation accuracy
        ax2 = ax1.twinx()
        ax2.plot(accuracies, color='tab:orange', label='Validation Accuracy')
        ax2.set_ylabel('Validation Accuracy', color='tab:orange')
        ax2.tick_params(axis='y', labelcolor='tab:orange')

        # Title and grid
        plt.title('Training Loss and Validation Accuracy over Epochs')
        ax1.grid(True)

        # Save the plot as a JPEG file
        plt.tight_layout()
        plt.savefig(f'{args.model}_plot_{args.seed}.jpg', format='jpeg')

        # Show the plot
        plt.show()