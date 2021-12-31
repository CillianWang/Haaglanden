from numpy.core.numeric import zeros_like
import torch
import numpy as np
import torch.nn as nn
import torchbnn as bnn
import matplotlib.pyplot as plt

def out_5Dvariance(input_tensor):
    
    input_tensor = torch.exp(input_tensor)
    shape = input_tensor.shape
    batch_size = shape[1]
    
    
    if shape[-1] != 5:
        print("error: should have output of 5 dimension")
    # compute the variance of all votes
    mean = input_tensor.mean(0)
    
    distance = torch.Tensor(np.zeros(batch_size))
    for vote in range(shape[0]):
        distance += ((input_tensor[vote]-mean)**2).sum(dim=-1)
        
        
    return distance


def calc_entropy_mean(input_tensor):
    # As the last layers is already logsoftmax:
    
    
    probs = input_tensor
    p_log_p = torch.log(probs) * probs
    entropy = -p_log_p.mean()
    if entropy < 0:
        print('now')
    return entropy




def train_bnn_past_and_future(model, optimizer, dataloader, val_dataloader, batch_size, epochs, gamma, block, divide, kl_weight = 0.1, kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)):
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma)
    for epoch in range(epochs):
        conf_matrix = np.zeros([5,5])
        correct = 0
        count = 0
        total_loss = 0
        index = 0
        
        
        for dl, rl in dataloader:
            if dl.shape[0] != batch_size:
                continue

            batch_correct = 0
            dl, rl = dl.cuda(), rl.cuda()
            # if rl.max()==5:
            #     continue
            optimizer.zero_grad()


            output = model(dl)

            if output.shape[0] != batch_size:
                continue
        
            # samples_per_cls = count_class(rl, 5)
            # loss = CB_loss(rl.reshape(batch_size,).cpu(), output.cpu(), samples_per_cls, 5, "focal", 0.9999, 2)
            loss = nn.CrossEntropyLoss()(output.reshape(batch_size, 5, 1), rl.long())
            kl = kl_loss(model)
            # rl = F.one_hot(rl.to(torch.int64), 5)\
            loss = loss + kl_weight*kl
            loss.backward()
            optimizer.step()
            # print(str(epoch)+":"+"loss equals to"+str(float(loss)))

            total_loss += loss


            estimate = output.argmax(-1)
            count += batch_size
            for i in range(batch_size):
                conf_matrix[int(estimate[i])][int(rl[i])]+=1
                if int(estimate[i])==int(rl[i]):
                    correct += 1
                    batch_correct +=1
            
            

        acc = correct/count


        # validation:
        conf_matrix_val = np.zeros([5,5])
        correct_val = 0
        count_val = 0
        total_loss_val = 0
        total_uncertainty = 0

        certainty = 0
        uncertainty = 0
        almost_certain = 0
        vague = 0

        certainty_correct = 0
        uncertainty_correct = 0
        almost_certain_correct = 0
        vague_correct = 0

        certainty_e_var = 0
        uncertainty_e_var = 0
        almost_certain_e_var = 0
        vague_e_var = 0
        
        certainty_e_mean = 0
        uncertainty_e_mean = 0
        almost_certain_e_mean = 0
        vague_e_mean = 0
        
        index = 0
        
        block_mean = 0
        block_var  = 0
        block_correct = 0
        block_count = 0
        
        total_out_var = torch.Tensor()
        total_rl = torch.Tensor()
        total_estimate = torch.Tensor()
        
        
        for dl, rl in val_dataloader:

            if dl.shape[0] != batch_size:
                continue
            
            batch_correct = 0
            dl, rl = dl.cuda(), rl.cuda()
            # if rl.max()==5:
            #     continue
            optimizer.zero_grad()
            



            batch_correct_val = 0
            if epoch> -1:
                out_vote = torch.zeros(10, batch_size)
                out_var = torch.zeros(10, batch_size, 5)
                for i in range(10):
                    with torch.no_grad():
                        out = model(dl)
                        out_vote[i] = out.argmax(-1)
                        out_var[i] = out.cpu()
                        
                
                out_entropy_mean = torch.exp(out_var).mean(0)
                out_entropy_var = out_5Dvariance(out_var)
                
                estimate = out_var.mean(0).argmax(-1)
                for i in range(batch_size):
                    if int(rl[i]) == block:
                        block_count += 1
                        block_mean += calc_entropy_mean(out_entropy_mean[i])
                        block_var += out_entropy_var[i].mean()
                        if int(estimate[i]) == block:
                            block_correct += 1
                        
                    if out_vote[:, i].max() != out_vote[:, i].min():
                        dice_count = np.zeros(5)
                        for dice in out_vote[:,i]:
                            dice_count[int(dice)] += 1

                        if dice_count.max() < 5:
                            uncertainty += 1
                            if int(out_vote[:,i].max()) == int(rl[i]):
                                uncertainty_correct += 1
                            uncertainty_e_mean += calc_entropy_mean(out_entropy_mean[i])
                            uncertainty_e_var +=  out_entropy_var[i].mean()
                                

                        elif dice_count.max() > 8:
                            almost_certain += 1    
                            if int(out_vote[:,i].max()) == int(rl[i]):
                                almost_certain_correct += 1
                            almost_certain_e_mean += calc_entropy_mean(out_entropy_mean[i])
                            almost_certain_e_var +=  out_entropy_var[i].mean()
                        else:
                            vague += 1
                            if int(out_vote[:,i].max()) == int(rl[i]):
                                vague_correct += 1
                            vague_e_mean += calc_entropy_mean(out_entropy_mean[i])
                            vague_e_var +=  out_entropy_var[i].mean()
                    else:
                        certainty += 1
                        if int(out_vote[:,i].max()) == int(rl[i]):
                            certainty_correct += 1
                        certainty_e_mean += calc_entropy_mean(out_entropy_mean[i])
                        certainty_e_var +=  out_entropy_var[i].mean()
                        
            with torch.no_grad():
                output = model(dl)


            total_out_var = torch.cat((total_out_var,out_entropy_var), dim=0)
            total_rl = torch.cat((total_rl, rl.cpu()), dim=0)
            total_estimate = torch.cat((total_estimate, estimate), dim=0)

            if output.shape[0] != batch_size:
                continue

            loss = nn.CrossEntropyLoss()(output.reshape(batch_size, 5, 1), rl.long())
            kl = kl_loss(model)
            # rl = F.one_hot(rl.to(torch.int64), 5)\
            loss = loss + kl_weight*kl
            # print(str(epoch)+":"+"loss equals to"+str(float(loss)))

            total_loss_val += loss

            count_val += batch_size
            for i in range(batch_size):
                conf_matrix_val[int(estimate[i])][int(rl[i])]+=1
                if int(estimate[i])==int(rl[i]):
                    correct_val += 1
                    batch_correct_val +=1
            

        acc_val = correct_val/count_val             

        def certainty_acc(correct, total_num):
            if total_num == 0:
                return 0
            else:
                return correct/total_num

        # plot acc vs variance
        var_sort = np.sort(total_out_var)
        rank = np.argsort(total_out_var)
        var_min = total_out_var.min()
        var_max = total_out_var.max()
        getting = np.linspace(0, var_sort.shape[0], divide).round()
        getting[-1] = getting[-1] - 1
        getting_vars = torch.Tensor(var_sort)[getting]
        
        #groups = np.linspace(var_min, var_max, divide)
        groups = getting_vars
        group_index = 1
        shreshold = groups[group_index]
        group_acc = np.zeros(divide-1)
        group_correct = 0
        group_count = 0
        
        
        variance_divide = np.linspace(var_min, var_max, divide)
        shreshold_index = 1
        shreshold_acc = np.zeros(divide-1)
        shreshold_leftout = np.zeros(divide-1)
        shreshold_correct = 0
        shreshold_count = 0
        shreshold_count_list = np.zeros(divide-1)
        for i in rank:
            if total_out_var[i] >= variance_divide[shreshold_index]:
                shreshold_count_list[shreshold_index-1] = shreshold_count
                shreshold_acc[shreshold_index-1] = certainty_acc(shreshold_correct, shreshold_count)

                rank_left_out = rank[shreshold_count:]
                shreshold_leftout_correct = 0
                for n in rank_left_out:
                    if int(total_estimate[n]) == int(total_rl[n]):
                        shreshold_leftout_correct += 1
                shreshold_leftout[shreshold_index-1] = certainty_acc(shreshold_leftout_correct, (rank.shape[0]-shreshold_count))
                shreshold_index += 1
                
            if total_out_var[i] == var_max:
                continue
            shreshold_count += 1
            if int(total_estimate[i]) == int(total_rl[i]):
                shreshold_correct += 1
        
        
        if epoch%15 ==0:    
            end_point = shreshold_count_list.argmax()        
            fig, ax1 = plt.subplots()
            color = 'tab:red'
            ax1.set_xlabel('Variance')
            ax1.set_ylabel('Accuracy', color=color)
            ax1.plot(variance_divide[1:end_point+1], shreshold_acc[:end_point], color=color)
            ax1.plot(variance_divide[1:end_point+1], shreshold_leftout[:end_point], color='tab:green')
            ax1.legend(['Accuracy','Left out accuracy'])
            ax1.tick_params(axis='y', labelcolor=color)

            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

            color = 'tab:blue'
            ax2.set_ylabel('Data amount', color=color)  # we already handled the x-label with ax1
            ax2.plot(variance_divide[1:end_point+1], shreshold_count_list[:end_point]/rank.shape[0], color=color)
            ax2.legend(['Amount of data'])
            ax2.tick_params(axis='y', labelcolor=color)

            fig.tight_layout()  # otherwise the right y-label is slightly clipped
            plt.savefig('acc_vs_var_shreshold'+ str(epoch)+'.png')
        
        
        for i in rank:
            if total_out_var[i] >= shreshold:
                group_acc[group_index-1] = certainty_acc(group_correct, group_count)
                group_index += 1
                if total_out_var[i] == var_max:
                    continue
                else:
                    shreshold = groups[group_index]
                group_correct = 0
                group_count = 0
                
            group_count += 1
            if int(total_estimate[i]) == int(total_rl[i]):
                group_correct += 1
        
        
        index_50 = 0
        index_75 = 0
        index_25 = 0
        for i in groups:
            if i > groups.mean():
                break
            index_50 += 1
            
        for i in groups[:index_50]:
            if i > groups[:index_50].mean():
                break
            index_75 += 1
            
        for i in groups[index_50:]:
            if i > groups[index_50:].mean():
                break
            index_25 += 1
        groups_mean = index_50
        groups_25 = index_25+index_50
        groups_75 = index_75
        print("There are "+ str(groups_25/divide*100)+"%" +" below 3/4 entropy, and their acc: "+str(group_acc[:groups_25].mean()))
        print("There are "+ str(index_50/divide*100)+"%" +" below average entropy, and their acc: "+str(group_acc[:index_50].mean()))
        print("There are "+ str(groups_75/divide*100)+"%" +" below 1/4 entropy, and their acc: "+str(group_acc[:groups_75].mean()))
        
        
        # An idea of taking only the 1/4 entropy results as serious:
        serious = rank[:round(groups_75/divide*rank.shape[0])]
        serious_estimate = total_estimate[serious]
        serious_scores = np.zeros(5)
        for i in serious_estimate:
            for n in range(5):
                if int(i) == n:
                    serious_scores[n]+=1
        print(serious_scores)
        serious_hist = np.zeros(rank.shape[0])
        for i in serious:
            serious_hist[i] = 1
            
        # see if data near serious estimation data can improve their accuracy:
        improved_estimate = total_estimate
        for i in range(rank.shape[0]):
            if i in serious:
                continue
            if (i-1) in serious:
                improved_estimate[i] == total_estimate[i-1]
                continue
            if (i+1) in serious:
                improved_estimate[i] == total_estimate[i+1]
        improved_correct = 0
        for i in range(rank.shape[0]):
            if int(improved_estimate[i]) == int(total_rl[i]):
                improved_correct += 1
        print("Improved acc: "+str(improved_correct/rank.shape[0]))
        
                
            
            
        
            
        if epoch%15 == 0:
            #fig = plt.figure()
            fig, ax = plt.subplots()
            color = 'tab:red'
            ax.set_xlabel('Variance')
            ax.set_ylabel('Accuracy', color=color)
            plt.scatter(groups[1:divide], group_acc)
            plt.savefig('acc_vs_var_'+str(epoch/15)+'.png')

            #see when acc goes to zero:
            for i in range(divide):
                if group_acc[-i-1] == 0:
                    continue
                else:
                    break
            zero_point = i
            print("There are "+str(round(i*total_estimate.shape[0]/divide))+" data with too large variance to get 0 acc, taking " + str(100*zero_point/divide)+"%")
        
            point = []
            for i in range(round(index_25/divide*rank.shape[0])):
                point.append(int(rank[-i-1]))

            # confusion matrix of esitmation of high variance
            hv_estimate = total_estimate[point]
            hv_rl = total_rl[point]
            hv_len = zero_point
            hv_confusion = np.zeros([5,5])
            
            for i in range(hv_len):
                hv_confusion[int(hv_estimate[i])][int(hv_rl[i])] +=1
            print("Highest variance dta's confusion matrix, estimation vs truth: ")
            print(hv_confusion)

            # others'acc:
            print("Remove high variance data, accuracy will be "+ str(group_acc[:(group_acc.shape[0]-zero_point)].mean()))
        
        

        scheduler.step()
        print("epoch:" +str(epoch)+"         Training loss:" + str(round(float(total_loss),1)) + "  acc:"+str(round(acc,3)) +
                         "       ;  Validation loss:" + str(round(float(total_loss_val),1))+ "  acc:"+ str(round(acc_val,3)) +
                          " ;uncertain:" + str(uncertainty)+ " ;vauge:"+str(vague)+ " ;almost:"+str(almost_certain)+ " ;certain:"+str(certainty))
        
        print("uncertain acc: " + str(certainty_acc(uncertainty_correct,uncertainty))) 
        print("vague acc:     " + str(certainty_acc(vague_correct,vague)))
        print("almost acc:    " + str(certainty_acc(almost_certain_correct,almost_certain)))
        print("certain acc:   " + str(certainty_acc(certainty_correct,certainty)))
        print("")
        print("Uncertainty means'entropy     : " + str(certainty_acc(uncertainty_e_mean, uncertainty)))
        print("Vague means'entropy           : " + str(certainty_acc(vague_e_mean, vague)))
        print("almost_certainty means'entropy: " + str(certainty_acc(almost_certain_e_mean, almost_certain)))
        print("certainty means'entropy       : " + str(certainty_acc(certainty_e_mean, certainty)))
        
        print("")
        print("Uncertainty var               : " + str(certainty_acc(uncertainty_e_var, uncertainty)))
        print("Vague var                     : " + str(certainty_acc(vague_e_var, vague)))
        print("almost_certainty var          : " + str(certainty_acc(almost_certain_e_var, almost_certain)))
        print("certainty var                 : " + str(certainty_acc(certainty_e_var, certainty)))
        
        print("")
        print("blocked class's acc:              :" + str(certainty_acc(block_correct,block_count)))
        print("blocked class's mean total entropy:" + str(certainty_acc(block_mean, block_count)))
        print("blocked class's var               :" + str(certainty_acc(block_var, block_count)))
        print("")
        
        
        if epoch%15 == 0:
            print("After 15 epochs, Training Confusion matrix:  (estimation vs truth)")
            print(conf_matrix)
        if epoch%15 == 0:
            print("After 15 epochs, Validation Confusion matrix:(estimation vs truth)")
            print(conf_matrix_val)
        print("")
        
        
        if epoch == 300:
            torch.save(model.state_dict(), 'Para_DeepNet_BNN') 