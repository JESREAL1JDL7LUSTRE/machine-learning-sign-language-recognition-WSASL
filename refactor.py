import re

def process_file(filepath, out_json, out_pth, early_fusion=True, is_3stream=False):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Imports
    content = content.replace("import os, sys", "import os, sys, json")
    
    # 2. Paths
    content = content.replace('MODEL_SAVE    = os.path.join(ROOT, "models", "sign_stgcn.pth")',
                              f'MODEL_SAVE    = os.path.join(ROOT, "models", "{out_pth}")')
    content = re.sub(r'EARLY_FUSION\s*=\s*True', f'EARLY_FUSION      = {early_fusion}', content)

    # 3. Eval model update
    eval_orig = """def eval_model(m, val_loader):
    m.eval()
    correct = 0
    with torch.no_grad():
        for xj, xmj, xb, xbm, yb in val_loader:
            xj, xmj, xb, xbm, yb = (xj.to(DEVICE), xmj.to(DEVICE), xb.to(DEVICE),
                                    xbm.to(DEVICE), yb.to(DEVICE))
            correct += (m(xj, motion=xmj, bone=xb, bone_motion=xbm)
                        .argmax(1) == yb).sum().item()
    return correct / len(val_loader.dataset)"""

    eval_new_4stream = """def eval_model(m, val_loader):
    m.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for xj, xmj, xb, xbm, yb in val_loader:
            xj, xmj, xb, xbm, yb = (xj.to(DEVICE), xmj.to(DEVICE), xb.to(DEVICE),
                                    xbm.to(DEVICE), yb.to(DEVICE))
            preds = m(xj, motion=xmj, bone=xb, bone_motion=xbm).argmax(1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(yb.cpu().numpy().tolist())
    correct = sum(1 for p, y in zip(all_preds, all_labels) if p == y)
    return correct / len(val_loader.dataset), all_preds, all_labels"""

    eval_new_3stream = """def eval_model(m, val_loader):
    m.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for xj, xmj, xb, yb in val_loader:
            xj, xmj, xb, yb = (xj.to(DEVICE), xmj.to(DEVICE), xb.to(DEVICE), yb.to(DEVICE))
            preds = m(xj, motion=xmj, bone=xb).argmax(1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(yb.cpu().numpy().tolist())
    correct = sum(1 for p, y in zip(all_preds, all_labels) if p == y)
    return correct / len(val_loader.dataset), all_preds, all_labels"""

    if is_3stream:
        content = content.replace(eval_orig, eval_new_3stream)
    else:
        content = content.replace(eval_orig, eval_new_4stream)

    # 4. train_fold update
    content = content.replace("v_acc = eval_model(model, val_loader)", "v_acc, vp, vl = eval_model(model, val_loader)")
    content = content.replace("v_swa = eval_model(swa_model, val_loader)", "v_swa, vp_swa, vl_swa = eval_model(swa_model, val_loader)")
    
    content = content.replace("best_state    = None", "best_state    = None\n    best_vp = []\n    best_vl = []\n    history = {'train_acc': [], 'val_acc': []}")
    
    content = re.sub(r'tr_acc = tr_correct / len\(train_loader.dataset\)',
                     r'tr_acc = tr_correct / len(train_loader.dataset)\n        history["train_acc"].append(tr_acc)', content)
    
    content = re.sub(r'if v_acc > best_val_acc:\n\s*best_val_acc = v_acc\n\s*best_state   = \{',
                     r'history["val_acc"].append(v_acc)\n        if v_acc > best_val_acc:\n            best_val_acc = v_acc\n            best_vp = vp\n            best_vl = vl\n            best_state   = {', content)
    
    content = content.replace("best_val_acc = v_swa\n            best_state = {", "best_val_acc = v_swa\n            best_vp = vp_swa\n            best_vl = vl_swa\n            best_state = {")
    content = content.replace("return best_val_acc, best_state", "return best_val_acc, best_state, best_vp, best_vl, history")

    # 5. train() update
    content = content.replace("fold_accs = []\n    best_acc  = 0.0\n    best_state = None",
                              "fold_accs = []\n    best_acc  = 0.0\n    best_state = None\n    cv_preds = []\n    cv_labels = []\n    fold_histories = []")

    content = content.replace("val_acc, state = train_fold(", "val_acc, state, vp, vl, hist = train_fold(")
    content = content.replace("fold_accs.append(val_acc)", "fold_accs.append(val_acc)\n        cv_preds.extend(vp)\n        cv_labels.extend(vl)\n        fold_histories.append(hist)")

    # 6. Test eval update
    test_eval_orig = """    tc = 0
    model.eval()
    with torch.no_grad():
        for xj, xmj, xb, xbm, yb in test_loader:
            xj, xmj, xb, xbm, yb = (xj.to(DEVICE), xmj.to(DEVICE), xb.to(DEVICE),
                                    xbm.to(DEVICE), yb.to(DEVICE))
            tc += (model(xj, motion=xmj, bone=xb, bone_motion=xbm)
                   .argmax(1) == yb).sum().item()"""
    
    test_eval_new_4stream = """    model.eval()
    test_preds = []
    test_labels = []
    with torch.no_grad():
        for xj, xmj, xb, xbm, yb in test_loader:
            xj, xmj, xb, xbm, yb = (xj.to(DEVICE), xmj.to(DEVICE), xb.to(DEVICE), xbm.to(DEVICE), yb.to(DEVICE))
            preds = model(xj, motion=xmj, bone=xb, bone_motion=xbm).argmax(1)
            test_preds.extend(preds.cpu().numpy().tolist())
            test_labels.extend(yb.cpu().numpy().tolist())
    tc = sum(1 for p, y in zip(test_preds, test_labels) if p == y)"""
    
    test_eval_new_3stream = """    model.eval()
    test_preds = []
    test_labels = []
    with torch.no_grad():
        for xj, xmj, xb, yb in test_loader:
            xj, xmj, xb, yb = (xj.to(DEVICE), xmj.to(DEVICE), xb.to(DEVICE), yb.to(DEVICE))
            preds = model(xj, motion=xmj, bone=xb).argmax(1)
            test_preds.extend(preds.cpu().numpy().tolist())
            test_labels.extend(yb.cpu().numpy().tolist())
    tc = sum(1 for p, y in zip(test_preds, test_labels) if p == y)"""

    if is_3stream:
        content = content.replace(test_eval_orig, test_eval_new_3stream)
    else:
        content = content.replace(test_eval_orig, test_eval_new_4stream)

    # 7. JSON output
    json_save = f"""    print(f"   Model saved   : {{MODEL_SAVE}}")

    # Load label map
    lmap_path = os.path.join(OUTPUT_DIR, "label_map.json")
    if os.path.exists(lmap_path):
        with open(lmap_path, 'r') as f:
            lmap = json.load(f)
    else:
        lmap = {{str(i): i for i in range(num_classes)}}

    results = {{
        "fold_accs": fold_accs,
        "cv_mean": float(np.mean(fold_accs)),
        "cv_std": float(np.std(fold_accs)),
        "test_acc": float(tc/len(y_test)),
        "all_preds": test_preds,
        "all_labels": test_labels,
        "cv_preds": cv_preds,
        "cv_labels": cv_labels,
        "num_classes": num_classes,
        "fold_histories": fold_histories,
        "label_map": lmap
    }}
    
    out_json = os.path.join(OUTPUT_DIR, "{out_json}")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"   JSON saved    : {{out_json}}")"""

    content = content.replace('    print(f"   Model saved   : {MODEL_SAVE}")', json_save)

    # If 3-stream, more replacements
    if is_3stream:
        # Import ThreeStream model
        content = content.replace("from models.st_gcn_twostream import Model as FourStreamSTGCN", "from models.st_gcn_twostream import Model as ThreeStreamSTGCN")
        content = content.replace("FourStreamSTGCN", "ThreeStreamSTGCN")
        content = content.replace("FourStreamDataset", "ThreeStreamDataset")
        
        # Dataset definition replacement
        content = re.sub(r'class ThreeStreamDataset\(Dataset\):.*?def __init__\(self, X_joint, X_motion, X_bone, X_bone_motion, y, augment=False\):',
                         r'class ThreeStreamDataset(Dataset):\n    def __init__(self, X_joint, X_motion, X_bone, y, augment=False):', content, flags=re.DOTALL)
        content = content.replace("self.X_bm    = X_bone_motion.astype(np.float32)", "")
        content = content.replace("xbm = self.X_bm[idx].copy()", "")
        content = content.replace("xj, xmj, xb, xbm = augment_4streams(xj, xmj, xb, xbm)", "xj, xmj, xb = augment_streams(xj, xmj, xb)")
        # Make sure augment_4streams def becomes augment_streams
        content = content.replace("def augment_4streams(xj, xm, xb, xbm):", "def augment_streams(xj, xm, xb):")
        content = content.replace("xbm = xbm + np.random.randn(*xbm.shape).astype(np.float32) * 0.015", "")
        content = content.replace("xbm = xbm * scale", "")
        content = content.replace("xbm = apply_rot(xbm)", "")
        content = content.replace("xbm[:, 0::2] = -xbm[:, 0::2]", "")
        content = content.replace("xbm = xbm[mask][idx]", "")
        content = content.replace("xbm = xbm[idx]", "")
        content = content.replace("xbm = xbm[st:st+cl][idx]", "")
        content = content.replace("xbm2 = xbm.reshape(T, V, 2)", "")
        content = content.replace("xbm2[i] = (rot @ xbm2[i].T).T", "")
        content = content.replace("xbm = xbm2.reshape(T, F)", "")
        content = content.replace("return xj.astype(np.float32), xm.astype(np.float32), xb.astype(np.float32), xbm.astype(np.float32)", "return xj.astype(np.float32), xm.astype(np.float32), xb.astype(np.float32)")
        
        content = content.replace("torch.tensor(self.to_graph(xbm), dtype=torch.float32),", "")

        # Dataset calls
        content = content.replace("Xj_tr, Xm_tr, Xb_tr, Xbm_tr, y_tr", "Xj_tr, Xm_tr, Xb_tr, y_tr")
        content = content.replace("Xj_val, Xm_val, Xb_val, Xbm_val, y_val", "Xj_val, Xm_val, Xb_val, y_val")
        content = content.replace("Xj_tr, Xm_tr, Xb_tr, Xbm_tr, y_tr, augment=True", "Xj_tr, Xm_tr, Xb_tr, y_tr, augment=True")
        content = content.replace("Xj_val, Xm_val, Xb_val, Xbm_val, y_val, augment=False", "Xj_val, Xm_val, Xb_val, y_val, augment=False")
        content = content.replace("for xj, xmj, xb, xbm, yb in train_loader:", "for xj, xmj, xb, yb in train_loader:")
        content = content.replace("xj, xmj, xb, xbm, yb = (xj.to(DEVICE), xmj.to(DEVICE), xb.to(DEVICE),\n                                    xbm.to(DEVICE), yb.to(DEVICE))", 
                                  "xj, xmj, xb, yb = (xj.to(DEVICE), xmj.to(DEVICE), xb.to(DEVICE), yb.to(DEVICE))")
        content = content.replace("out = model(xj, motion=xmj, bone=xb, bone_motion=xbm)", "out = model(xj, motion=xmj, bone=xb)")
        
        # global_normalize calls
        content = content.replace("Xbm_tr_n, Xbm_val_n, Xbm_test_n = global_normalize(\n            Xbm_tv[tr_idx], Xbm_tv[val_idx], Xbm_test)", "")
        content = content.replace("Xj_tr_n, Xm_tr_n, Xb_tr_n, Xbm_tr_n, y_tv[tr_idx],\n            Xj_val_n, Xm_val_n, Xb_val_n, Xbm_val_n, y_tv[val_idx],",
                                  "Xj_tr_n, Xm_tr_n, Xb_tr_n, y_tv[tr_idx],\n            Xj_val_n, Xm_val_n, Xb_val_n, y_tv[val_idx],")
        content = content.replace("best_Xbm_test = Xbm_test_n", "")
        content = content.replace("best_Xbm_test = Xbm_test_n", "") # just in case
        content = content.replace("Xbm_test = X_bm[idx_test];", "")
        content = content.replace("Xbm_tv   = X_bm[idx_tv];", "")
        content = content.replace("X_joint, X_motion, X_bone, X_bm, y = load_data()", "X_joint, X_motion, X_bone, X_bm, y = load_data()") # wait, load_data returns 5
        
        content = content.replace("ThreeStreamDataset(best_Xj_test, best_Xm_test, best_Xb_test, best_Xbm_test,\n                          y_test, augment=False)",
                                  "ThreeStreamDataset(best_Xj_test, best_Xm_test, best_Xb_test, y_test, augment=False)")

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

process_file('training/train_4stream_early.py', 'results_4stream_early.json', 'sign_stgcn_4stream_early.pth', early_fusion=True)
process_file('training/train_4stream_late.py', 'results_4stream_late.json', 'sign_stgcn_4stream_late.pth', early_fusion=False)
process_file('training/train_3stream.py', 'results_3stream.json', 'sign_stgcn_3stream.pth', is_3stream=True)
