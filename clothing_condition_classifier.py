<html>
<head>
<title>clothing_condition_classifier.py</title>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8">
<style type="text/css">
.s0 { color: #cf8e6d;}
.s1 { color: #bcbec4;}
.s2 { color: #bcbec4;}
.s3 { color: #7a7e85;}
.s4 { color: #2aacb8;}
.s5 { color: #6aab73;}
</style>
</head>
<body bgcolor="#1e1f22">
<table CELLSPACING=0 CELLPADDING=5 COLS=1 WIDTH="100%" BGCOLOR="#606060" >
<tr><td><center>
<font face="Arial, Helvetica" color="#000000">
clothing_condition_classifier.py</font>
</center></td></tr></table>
<pre><span class="s0">import </span><span class="s1">os</span>
<span class="s0">import </span><span class="s1">random</span>
<span class="s0">import </span><span class="s1">torch</span>
<span class="s0">import </span><span class="s1">torch</span><span class="s2">.</span><span class="s1">nn </span><span class="s0">as </span><span class="s1">nn</span>
<span class="s0">import </span><span class="s1">torchvision</span><span class="s2">.</span><span class="s1">transforms </span><span class="s0">as </span><span class="s1">transforms</span>
<span class="s0">import </span><span class="s1">matplotlib</span><span class="s2">.</span><span class="s1">pyplot </span><span class="s0">as </span><span class="s1">plt</span>
<span class="s0">from </span><span class="s1">torchvision </span><span class="s0">import </span><span class="s1">models</span><span class="s2">, </span><span class="s1">datasets</span>
<span class="s0">from </span><span class="s1">torchvision</span><span class="s2">.</span><span class="s1">models </span><span class="s0">import </span><span class="s1">ResNet50_Weights</span>
<span class="s0">from </span><span class="s1">torch</span><span class="s2">.</span><span class="s1">utils</span><span class="s2">.</span><span class="s1">data </span><span class="s0">import </span><span class="s1">DataLoader</span><span class="s2">, </span><span class="s1">Subset</span>

<span class="s3"># Important for replicability during data splitting and subsequent training</span>
<span class="s1">SEED </span><span class="s2">= </span><span class="s4">42</span>
<span class="s1">random</span><span class="s2">.</span><span class="s1">seed</span><span class="s2">(</span><span class="s1">SEED</span><span class="s2">)</span>
<span class="s1">torch</span><span class="s2">.</span><span class="s1">manual_seed</span><span class="s2">(</span><span class="s1">SEED</span><span class="s2">)</span>

<span class="s1">data_set </span><span class="s2">= </span><span class="s5">r&quot;C:\Users\admin\Desktop\clothes_dataset&quot;</span>

<span class="s3"># Check if dataset exists</span>
<span class="s0">if not </span><span class="s1">os</span><span class="s2">.</span><span class="s1">path</span><span class="s2">.</span><span class="s1">exists</span><span class="s2">(</span><span class="s1">data_set</span><span class="s2">):</span>
    <span class="s0">raise </span><span class="s1">FileNotFoundError</span><span class="s2">(</span><span class="s5">f&quot;Dataset path does not exist: </span><span class="s0">{</span><span class="s1">data_set</span><span class="s0">}</span><span class="s5">&quot;</span><span class="s2">)</span>

<span class="s3"># Check if dataset contains valid image files</span>
<span class="s0">for </span><span class="s1">class_folder </span><span class="s0">in </span><span class="s1">os</span><span class="s2">.</span><span class="s1">listdir</span><span class="s2">(</span><span class="s1">data_set</span><span class="s2">):</span>
    <span class="s1">class_path </span><span class="s2">= </span><span class="s1">os</span><span class="s2">.</span><span class="s1">path</span><span class="s2">.</span><span class="s1">join</span><span class="s2">(</span><span class="s1">data_set</span><span class="s2">, </span><span class="s1">class_folder</span><span class="s2">)</span>
    <span class="s0">if </span><span class="s1">os</span><span class="s2">.</span><span class="s1">path</span><span class="s2">.</span><span class="s1">isdir</span><span class="s2">(</span><span class="s1">class_path</span><span class="s2">):</span>
        <span class="s1">images </span><span class="s2">= [</span><span class="s1">f </span><span class="s0">for </span><span class="s1">f </span><span class="s0">in </span><span class="s1">os</span><span class="s2">.</span><span class="s1">listdir</span><span class="s2">(</span><span class="s1">class_path</span><span class="s2">) </span><span class="s0">if </span><span class="s1">f</span><span class="s2">.</span><span class="s1">lower</span><span class="s2">().</span><span class="s1">endswith</span><span class="s2">((</span><span class="s5">'.jpg'</span><span class="s2">, </span><span class="s5">'.jpeg'</span><span class="s2">, </span><span class="s5">'.png'</span><span class="s2">, </span><span class="s5">'.bmp'</span><span class="s2">, </span><span class="s5">'.tiff'</span><span class="s2">, </span><span class="s5">'.webp'</span><span class="s2">))]</span>
        <span class="s0">if not </span><span class="s1">images</span><span class="s2">:</span>
            <span class="s0">raise </span><span class="s1">FileNotFoundError</span><span class="s2">(</span><span class="s5">f&quot;No valid images found in: </span><span class="s0">{</span><span class="s1">class_path</span><span class="s0">}</span><span class="s5">&quot;</span><span class="s2">)</span>

<span class="s1">print</span><span class="s2">(</span><span class="s5">&quot;Dataset structure is valid. Proceeding with training...&quot;</span><span class="s2">)</span>

<span class="s3"># Define data augmentation &amp; normalization</span>
<span class="s1">transform </span><span class="s2">= </span><span class="s1">transforms</span><span class="s2">.</span><span class="s1">Compose</span><span class="s2">([</span>
    <span class="s1">transforms</span><span class="s2">.</span><span class="s1">Resize</span><span class="s2">(</span><span class="s4">256</span><span class="s2">),</span>
    <span class="s1">transforms</span><span class="s2">.</span><span class="s1">CenterCrop</span><span class="s2">(</span><span class="s4">224</span><span class="s2">),</span>
    <span class="s1">transforms</span><span class="s2">.</span><span class="s1">RandomHorizontalFlip</span><span class="s2">(),</span>
    <span class="s1">transforms</span><span class="s2">.</span><span class="s1">RandomRotation</span><span class="s2">(</span><span class="s4">15</span><span class="s2">),</span>
    <span class="s1">transforms</span><span class="s2">.</span><span class="s1">ToTensor</span><span class="s2">(),</span>
    <span class="s1">transforms</span><span class="s2">.</span><span class="s1">Normalize</span><span class="s2">(</span><span class="s1">mean</span><span class="s2">=[</span><span class="s4">0.485</span><span class="s2">, </span><span class="s4">0.456</span><span class="s2">, </span><span class="s4">0.406</span><span class="s2">], </span><span class="s1">std</span><span class="s2">=[</span><span class="s4">0.229</span><span class="s2">, </span><span class="s4">0.224</span><span class="s2">, </span><span class="s4">0.225</span><span class="s2">])</span>
<span class="s2">])</span>

<span class="s3"># Load dataset</span>
<span class="s1">data </span><span class="s2">= </span><span class="s1">datasets</span><span class="s2">.</span><span class="s1">ImageFolder</span><span class="s2">(</span><span class="s1">data_set</span><span class="s2">, </span><span class="s1">transform</span><span class="s2">=</span><span class="s1">transform</span><span class="s2">)</span>

<span class="s3"># Check dataset size before proceeding</span>
<span class="s1">print</span><span class="s2">(</span><span class="s5">f&quot;Total images in dataset: </span><span class="s0">{</span><span class="s1">len</span><span class="s2">(</span><span class="s1">data</span><span class="s2">)</span><span class="s0">}</span><span class="s5">&quot;</span><span class="s2">)</span>
<span class="s0">if </span><span class="s1">len</span><span class="s2">(</span><span class="s1">data</span><span class="s2">) == </span><span class="s4">0</span><span class="s2">:</span>
    <span class="s0">raise </span><span class="s1">ValueError</span><span class="s2">(</span><span class="s5">&quot;Dataset is empty. Ensure images are present in the class folders.&quot;</span><span class="s2">)</span>

<span class="s3"># Data split (70/15/15)</span>
<span class="s1">class_indices </span><span class="s2">= {</span><span class="s1">class_idx</span><span class="s2">: [] </span><span class="s0">for </span><span class="s1">class_idx </span><span class="s0">in </span><span class="s1">range</span><span class="s2">(</span><span class="s1">len</span><span class="s2">(</span><span class="s1">data</span><span class="s2">.</span><span class="s1">classes</span><span class="s2">))}</span>

<span class="s0">for </span><span class="s1">idx</span><span class="s2">, (</span><span class="s1">_</span><span class="s2">, </span><span class="s1">label</span><span class="s2">) </span><span class="s0">in </span><span class="s1">enumerate</span><span class="s2">(</span><span class="s1">data</span><span class="s2">.</span><span class="s1">samples</span><span class="s2">):</span>
    <span class="s1">class_indices</span><span class="s2">[</span><span class="s1">label</span><span class="s2">].</span><span class="s1">append</span><span class="s2">(</span><span class="s1">idx</span><span class="s2">)</span>

<span class="s1">train_indices</span><span class="s2">, </span><span class="s1">val_indices</span><span class="s2">, </span><span class="s1">test_indices </span><span class="s2">= [], [], []</span>

<span class="s0">for </span><span class="s1">class_idx</span><span class="s2">, </span><span class="s1">indices </span><span class="s0">in </span><span class="s1">class_indices</span><span class="s2">.</span><span class="s1">items</span><span class="s2">():</span>
    <span class="s1">random</span><span class="s2">.</span><span class="s1">shuffle</span><span class="s2">(</span><span class="s1">indices</span><span class="s2">)</span>
    <span class="s1">train_size </span><span class="s2">= </span><span class="s1">int</span><span class="s2">(</span><span class="s4">0.7 </span><span class="s2">* </span><span class="s1">len</span><span class="s2">(</span><span class="s1">indices</span><span class="s2">))</span>
    <span class="s1">val_size </span><span class="s2">= </span><span class="s1">int</span><span class="s2">(</span><span class="s4">0.15 </span><span class="s2">* </span><span class="s1">len</span><span class="s2">(</span><span class="s1">indices</span><span class="s2">))</span>
    <span class="s1">test_size </span><span class="s2">= </span><span class="s1">len</span><span class="s2">(</span><span class="s1">indices</span><span class="s2">) - </span><span class="s1">train_size </span><span class="s2">- </span><span class="s1">val_size</span>

    <span class="s1">train_indices</span><span class="s2">.</span><span class="s1">extend</span><span class="s2">(</span><span class="s1">indices</span><span class="s2">[:</span><span class="s1">train_size</span><span class="s2">])</span>
    <span class="s1">val_indices</span><span class="s2">.</span><span class="s1">extend</span><span class="s2">(</span><span class="s1">indices</span><span class="s2">[</span><span class="s1">train_size</span><span class="s2">:</span><span class="s1">train_size </span><span class="s2">+ </span><span class="s1">val_size</span><span class="s2">])</span>
    <span class="s1">test_indices</span><span class="s2">.</span><span class="s1">extend</span><span class="s2">(</span><span class="s1">indices</span><span class="s2">[</span><span class="s1">train_size </span><span class="s2">+ </span><span class="s1">val_size</span><span class="s2">:])</span>

<span class="s3"># Create subsets based on calculated indices</span>
<span class="s1">train_set </span><span class="s2">= </span><span class="s1">Subset</span><span class="s2">(</span><span class="s1">data</span><span class="s2">, </span><span class="s1">train_indices</span><span class="s2">)</span>
<span class="s1">val_set </span><span class="s2">= </span><span class="s1">Subset</span><span class="s2">(</span><span class="s1">data</span><span class="s2">, </span><span class="s1">val_indices</span><span class="s2">)</span>
<span class="s1">test_set </span><span class="s2">= </span><span class="s1">Subset</span><span class="s2">(</span><span class="s1">data</span><span class="s2">, </span><span class="s1">test_indices</span><span class="s2">)</span>

<span class="s3"># Print dataset split sizes</span>
<span class="s1">print</span><span class="s2">(</span><span class="s5">f&quot;Training samples: </span><span class="s0">{</span><span class="s1">len</span><span class="s2">(</span><span class="s1">train_set</span><span class="s2">)</span><span class="s0">}</span><span class="s5">, Validation samples: </span><span class="s0">{</span><span class="s1">len</span><span class="s2">(</span><span class="s1">val_set</span><span class="s2">)</span><span class="s0">}</span><span class="s5">, Test samples: </span><span class="s0">{</span><span class="s1">len</span><span class="s2">(</span><span class="s1">test_set</span><span class="s2">)</span><span class="s0">}</span><span class="s5">&quot;</span><span class="s2">)</span>

<span class="s1">batch_size </span><span class="s2">= </span><span class="s4">128  </span><span class="s3"># Increased batch size</span>

<span class="s3"># Set num_workers=0 to avoid DataLoader freezing issues</span>
<span class="s1">train_loader </span><span class="s2">= </span><span class="s1">DataLoader</span><span class="s2">(</span><span class="s1">train_set</span><span class="s2">, </span><span class="s1">batch_size</span><span class="s2">=</span><span class="s1">batch_size</span><span class="s2">, </span><span class="s1">shuffle</span><span class="s2">=</span><span class="s0">True</span><span class="s2">, </span><span class="s1">num_workers</span><span class="s2">=</span><span class="s4">0</span><span class="s2">)</span>
<span class="s1">val_loader </span><span class="s2">= </span><span class="s1">DataLoader</span><span class="s2">(</span><span class="s1">val_set</span><span class="s2">, </span><span class="s1">batch_size</span><span class="s2">=</span><span class="s1">batch_size</span><span class="s2">, </span><span class="s1">shuffle</span><span class="s2">=</span><span class="s0">False</span><span class="s2">, </span><span class="s1">num_workers</span><span class="s2">=</span><span class="s4">0</span><span class="s2">)</span>
<span class="s1">test_loader </span><span class="s2">= </span><span class="s1">DataLoader</span><span class="s2">(</span><span class="s1">test_set</span><span class="s2">, </span><span class="s1">batch_size</span><span class="s2">=</span><span class="s1">batch_size</span><span class="s2">, </span><span class="s1">shuffle</span><span class="s2">=</span><span class="s0">False</span><span class="s2">, </span><span class="s1">num_workers</span><span class="s2">=</span><span class="s4">0</span><span class="s2">)</span>

<span class="s3"># Detect if GPU is available and use it</span>
<span class="s1">device </span><span class="s2">= </span><span class="s1">torch</span><span class="s2">.</span><span class="s1">device</span><span class="s2">(</span><span class="s5">&quot;cuda&quot; </span><span class="s0">if </span><span class="s1">torch</span><span class="s2">.</span><span class="s1">cuda</span><span class="s2">.</span><span class="s1">is_available</span><span class="s2">() </span><span class="s0">else </span><span class="s5">&quot;cpu&quot;</span><span class="s2">)</span>
<span class="s1">print</span><span class="s2">(</span><span class="s5">f&quot;Using device: </span><span class="s0">{</span><span class="s1">device</span><span class="s0">}</span><span class="s5">&quot;</span><span class="s2">)</span>

<span class="s3"># Load pretrained ResNet50 model</span>
<span class="s1">model </span><span class="s2">= </span><span class="s1">models</span><span class="s2">.</span><span class="s1">resnet50</span><span class="s2">(</span><span class="s1">weights</span><span class="s2">=</span><span class="s1">ResNet50_Weights</span><span class="s2">.</span><span class="s1">IMAGENET1K_V1</span><span class="s2">)</span>

<span class="s3"># Modify last fully connected layer for 3 classes</span>
<span class="s1">num_ftrs </span><span class="s2">= </span><span class="s1">model</span><span class="s2">.</span><span class="s1">fc</span><span class="s2">.</span><span class="s1">in_features</span>
<span class="s1">model</span><span class="s2">.</span><span class="s1">fc </span><span class="s2">= </span><span class="s1">nn</span><span class="s2">.</span><span class="s1">Linear</span><span class="s2">(</span><span class="s1">num_ftrs</span><span class="s2">, </span><span class="s1">len</span><span class="s2">(</span><span class="s1">data</span><span class="s2">.</span><span class="s1">classes</span><span class="s2">))</span>

<span class="s1">model </span><span class="s2">= </span><span class="s1">model</span><span class="s2">.</span><span class="s1">to</span><span class="s2">(</span><span class="s1">device</span><span class="s2">)</span>

<span class="s3"># Define loss function and optimizer</span>
<span class="s1">criterion </span><span class="s2">= </span><span class="s1">nn</span><span class="s2">.</span><span class="s1">CrossEntropyLoss</span><span class="s2">()</span>
<span class="s1">optimizer </span><span class="s2">= </span><span class="s1">torch</span><span class="s2">.</span><span class="s1">optim</span><span class="s2">.</span><span class="s1">Adam</span><span class="s2">(</span><span class="s1">model</span><span class="s2">.</span><span class="s1">parameters</span><span class="s2">(), </span><span class="s1">lr</span><span class="s2">=</span><span class="s4">0.001</span><span class="s2">, </span><span class="s1">weight_decay</span><span class="s2">=</span><span class="s4">1e-4</span><span class="s2">)</span>

<span class="s3"># Training loop with loss &amp; accuracy tracking</span>
<span class="s1">num_epochs </span><span class="s2">= </span><span class="s4">5  </span><span class="s3"># Reduced epochs</span>
<span class="s1">train_losses</span><span class="s2">, </span><span class="s1">val_losses </span><span class="s2">= [], []</span>
<span class="s1">train_accuracies</span><span class="s2">, </span><span class="s1">val_accuracies </span><span class="s2">= [], []</span>

<span class="s1">best_val_loss </span><span class="s2">= </span><span class="s1">float</span><span class="s2">(</span><span class="s5">'inf'</span><span class="s2">)</span>
<span class="s1">epochs_without_improvement </span><span class="s2">= </span><span class="s4">0</span>
<span class="s1">patience </span><span class="s2">= </span><span class="s4">3  </span><span class="s3"># Stop training if no improvement for 3 epochs</span>

<span class="s0">for </span><span class="s1">epoch </span><span class="s0">in </span><span class="s1">range</span><span class="s2">(</span><span class="s1">num_epochs</span><span class="s2">):</span>
    <span class="s1">model</span><span class="s2">.</span><span class="s1">train</span><span class="s2">()</span>
    <span class="s1">running_loss</span><span class="s2">, </span><span class="s1">correct</span><span class="s2">, </span><span class="s1">total </span><span class="s2">= </span><span class="s4">0.0</span><span class="s2">, </span><span class="s4">0</span><span class="s2">, </span><span class="s4">0</span>

    <span class="s0">for </span><span class="s1">batch_idx</span><span class="s2">, (</span><span class="s1">images</span><span class="s2">, </span><span class="s1">labels</span><span class="s2">) </span><span class="s0">in </span><span class="s1">enumerate</span><span class="s2">(</span><span class="s1">train_loader</span><span class="s2">):</span>
        <span class="s1">images</span><span class="s2">, </span><span class="s1">labels </span><span class="s2">= </span><span class="s1">images</span><span class="s2">.</span><span class="s1">to</span><span class="s2">(</span><span class="s1">device</span><span class="s2">), </span><span class="s1">labels</span><span class="s2">.</span><span class="s1">to</span><span class="s2">(</span><span class="s1">device</span><span class="s2">)</span>

        <span class="s1">optimizer</span><span class="s2">.</span><span class="s1">zero_grad</span><span class="s2">()</span>
        <span class="s1">outputs </span><span class="s2">= </span><span class="s1">model</span><span class="s2">(</span><span class="s1">images</span><span class="s2">)</span>
        <span class="s1">loss </span><span class="s2">= </span><span class="s1">criterion</span><span class="s2">(</span><span class="s1">outputs</span><span class="s2">, </span><span class="s1">labels</span><span class="s2">)</span>
        <span class="s1">loss</span><span class="s2">.</span><span class="s1">backward</span><span class="s2">()</span>
        <span class="s1">optimizer</span><span class="s2">.</span><span class="s1">step</span><span class="s2">()</span>

        <span class="s1">running_loss </span><span class="s2">+= </span><span class="s1">loss</span><span class="s2">.</span><span class="s1">item</span><span class="s2">()</span>
        <span class="s1">_</span><span class="s2">, </span><span class="s1">predicted </span><span class="s2">= </span><span class="s1">outputs</span><span class="s2">.</span><span class="s1">max</span><span class="s2">(</span><span class="s4">1</span><span class="s2">)</span>
        <span class="s1">total </span><span class="s2">+= </span><span class="s1">labels</span><span class="s2">.</span><span class="s1">size</span><span class="s2">(</span><span class="s4">0</span><span class="s2">)</span>
        <span class="s1">correct </span><span class="s2">+= </span><span class="s1">predicted</span><span class="s2">.</span><span class="s1">eq</span><span class="s2">(</span><span class="s1">labels</span><span class="s2">).</span><span class="s1">sum</span><span class="s2">().</span><span class="s1">item</span><span class="s2">()</span>

        <span class="s1">batch_accuracy </span><span class="s2">= </span><span class="s4">100 </span><span class="s2">* </span><span class="s1">correct </span><span class="s2">/ </span><span class="s1">total</span>
        <span class="s0">if </span><span class="s1">batch_idx </span><span class="s2">% </span><span class="s4">5 </span><span class="s2">== </span><span class="s4">0</span><span class="s2">:</span>
            <span class="s1">print</span><span class="s2">(</span><span class="s5">f&quot;Epoch </span><span class="s0">{</span><span class="s1">epoch</span><span class="s2">+</span><span class="s4">1</span><span class="s0">} </span><span class="s5">| Batch </span><span class="s0">{</span><span class="s1">batch_idx</span><span class="s0">}</span><span class="s5">/</span><span class="s0">{</span><span class="s1">len</span><span class="s2">(</span><span class="s1">train_loader</span><span class="s2">)</span><span class="s0">} </span><span class="s5">- Loss: </span><span class="s0">{</span><span class="s1">loss</span><span class="s2">.</span><span class="s1">item</span><span class="s2">()</span><span class="s0">:</span><span class="s5">.4f</span><span class="s0">} </span><span class="s5">| Train Acc: </span><span class="s0">{</span><span class="s1">batch_accuracy</span><span class="s0">:</span><span class="s5">.2f</span><span class="s0">}</span><span class="s5">%&quot;</span><span class="s2">)</span>

    <span class="s1">train_accuracy </span><span class="s2">= </span><span class="s4">100 </span><span class="s2">* </span><span class="s1">correct </span><span class="s2">/ </span><span class="s1">total</span>
    <span class="s1">train_losses</span><span class="s2">.</span><span class="s1">append</span><span class="s2">(</span><span class="s1">running_loss </span><span class="s2">/ </span><span class="s1">len</span><span class="s2">(</span><span class="s1">train_loader</span><span class="s2">))</span>
    <span class="s1">train_accuracies</span><span class="s2">.</span><span class="s1">append</span><span class="s2">(</span><span class="s1">train_accuracy</span><span class="s2">)</span>

    <span class="s3"># Validation phase</span>
    <span class="s1">model</span><span class="s2">.</span><span class="s1">eval</span><span class="s2">()</span>
    <span class="s1">val_loss</span><span class="s2">, </span><span class="s1">val_correct</span><span class="s2">, </span><span class="s1">val_total </span><span class="s2">= </span><span class="s4">0.0</span><span class="s2">, </span><span class="s4">0</span><span class="s2">, </span><span class="s4">0</span>
    <span class="s0">with </span><span class="s1">torch</span><span class="s2">.</span><span class="s1">no_grad</span><span class="s2">():</span>
        <span class="s0">for </span><span class="s1">images</span><span class="s2">, </span><span class="s1">labels </span><span class="s0">in </span><span class="s1">val_loader</span><span class="s2">:</span>
            <span class="s1">images</span><span class="s2">, </span><span class="s1">labels </span><span class="s2">= </span><span class="s1">images</span><span class="s2">.</span><span class="s1">to</span><span class="s2">(</span><span class="s1">device</span><span class="s2">), </span><span class="s1">labels</span><span class="s2">.</span><span class="s1">to</span><span class="s2">(</span><span class="s1">device</span><span class="s2">)</span>

            <span class="s1">outputs </span><span class="s2">= </span><span class="s1">model</span><span class="s2">(</span><span class="s1">images</span><span class="s2">)</span>
            <span class="s1">loss </span><span class="s2">= </span><span class="s1">criterion</span><span class="s2">(</span><span class="s1">outputs</span><span class="s2">, </span><span class="s1">labels</span><span class="s2">)</span>

            <span class="s1">val_loss </span><span class="s2">+= </span><span class="s1">loss</span><span class="s2">.</span><span class="s1">item</span><span class="s2">()</span>
            <span class="s1">_</span><span class="s2">, </span><span class="s1">predicted </span><span class="s2">= </span><span class="s1">outputs</span><span class="s2">.</span><span class="s1">max</span><span class="s2">(</span><span class="s4">1</span><span class="s2">)</span>
            <span class="s1">val_total </span><span class="s2">+= </span><span class="s1">labels</span><span class="s2">.</span><span class="s1">size</span><span class="s2">(</span><span class="s4">0</span><span class="s2">)</span>
            <span class="s1">val_correct </span><span class="s2">+= </span><span class="s1">predicted</span><span class="s2">.</span><span class="s1">eq</span><span class="s2">(</span><span class="s1">labels</span><span class="s2">).</span><span class="s1">sum</span><span class="s2">().</span><span class="s1">item</span><span class="s2">()</span>

    <span class="s1">val_accuracy </span><span class="s2">= </span><span class="s4">100 </span><span class="s2">* </span><span class="s1">val_correct </span><span class="s2">/ </span><span class="s1">val_total</span>
    <span class="s1">val_losses</span><span class="s2">.</span><span class="s1">append</span><span class="s2">(</span><span class="s1">val_loss </span><span class="s2">/ </span><span class="s1">len</span><span class="s2">(</span><span class="s1">val_loader</span><span class="s2">))</span>
    <span class="s1">val_accuracies</span><span class="s2">.</span><span class="s1">append</span><span class="s2">(</span><span class="s1">val_accuracy</span><span class="s2">)</span>

    <span class="s1">print</span><span class="s2">(</span><span class="s5">f&quot;Epoch </span><span class="s0">{</span><span class="s1">epoch</span><span class="s2">+</span><span class="s4">1</span><span class="s0">}</span><span class="s5">/</span><span class="s0">{</span><span class="s1">num_epochs</span><span class="s0">} </span><span class="s5">| Train Acc: </span><span class="s0">{</span><span class="s1">train_accuracy</span><span class="s0">:</span><span class="s5">.2f</span><span class="s0">}</span><span class="s5">% | Val Acc: </span><span class="s0">{</span><span class="s1">val_accuracy</span><span class="s0">:</span><span class="s5">.2f</span><span class="s0">}</span><span class="s5">% | Val Loss: </span><span class="s0">{</span><span class="s1">val_loss</span><span class="s0">:</span><span class="s5">.4f</span><span class="s0">}</span><span class="s5">&quot;</span><span class="s2">)</span>

    <span class="s3"># Early stopping check</span>
    <span class="s0">if </span><span class="s1">val_loss </span><span class="s2">&lt; </span><span class="s1">best_val_loss</span><span class="s2">:</span>
        <span class="s1">best_val_loss </span><span class="s2">= </span><span class="s1">val_loss</span>
        <span class="s1">epochs_without_improvement </span><span class="s2">= </span><span class="s4">0</span>
    <span class="s0">else</span><span class="s2">:</span>
        <span class="s1">epochs_without_improvement </span><span class="s2">+= </span><span class="s4">1</span>
        <span class="s0">if </span><span class="s1">epochs_without_improvement </span><span class="s2">&gt;= </span><span class="s1">patience</span><span class="s2">:</span>
            <span class="s1">print</span><span class="s2">(</span><span class="s5">f&quot;Early stopping triggered at epoch </span><span class="s0">{</span><span class="s1">epoch</span><span class="s2">+</span><span class="s4">1</span><span class="s0">}</span><span class="s5">. Stopping training.&quot;</span><span class="s2">)</span>
            <span class="s0">break</span>

<span class="s3"># Evaluate on test set</span>
<span class="s1">model</span><span class="s2">.</span><span class="s1">eval</span><span class="s2">()</span>
<span class="s1">test_correct</span><span class="s2">, </span><span class="s1">test_total </span><span class="s2">= </span><span class="s4">0</span><span class="s2">, </span><span class="s4">0</span>

<span class="s0">with </span><span class="s1">torch</span><span class="s2">.</span><span class="s1">no_grad</span><span class="s2">():</span>
    <span class="s0">for </span><span class="s1">images</span><span class="s2">, </span><span class="s1">labels </span><span class="s0">in </span><span class="s1">test_loader</span><span class="s2">:</span>
        <span class="s1">images</span><span class="s2">, </span><span class="s1">labels </span><span class="s2">= </span><span class="s1">images</span><span class="s2">.</span><span class="s1">to</span><span class="s2">(</span><span class="s1">device</span><span class="s2">), </span><span class="s1">labels</span><span class="s2">.</span><span class="s1">to</span><span class="s2">(</span><span class="s1">device</span><span class="s2">)</span>
        <span class="s1">outputs </span><span class="s2">= </span><span class="s1">model</span><span class="s2">(</span><span class="s1">images</span><span class="s2">)</span>
        <span class="s1">_</span><span class="s2">, </span><span class="s1">predicted </span><span class="s2">= </span><span class="s1">outputs</span><span class="s2">.</span><span class="s1">max</span><span class="s2">(</span><span class="s4">1</span><span class="s2">)</span>
        <span class="s1">test_total </span><span class="s2">+= </span><span class="s1">labels</span><span class="s2">.</span><span class="s1">size</span><span class="s2">(</span><span class="s4">0</span><span class="s2">)</span>
        <span class="s1">test_correct </span><span class="s2">+= </span><span class="s1">predicted</span><span class="s2">.</span><span class="s1">eq</span><span class="s2">(</span><span class="s1">labels</span><span class="s2">).</span><span class="s1">sum</span><span class="s2">().</span><span class="s1">item</span><span class="s2">()</span>

<span class="s1">test_accuracy </span><span class="s2">= </span><span class="s4">100 </span><span class="s2">* </span><span class="s1">test_correct </span><span class="s2">/ </span><span class="s1">test_total</span>
<span class="s1">print</span><span class="s2">(</span><span class="s5">f&quot;Test Accuracy: </span><span class="s0">{</span><span class="s1">test_accuracy</span><span class="s0">:</span><span class="s5">.2f</span><span class="s0">}</span><span class="s5">%&quot;</span><span class="s2">)</span>

<span class="s3"># Plot Loss and Accuracy over Epochs</span>
<span class="s1">plt</span><span class="s2">.</span><span class="s1">figure</span><span class="s2">(</span><span class="s1">figsize</span><span class="s2">=(</span><span class="s4">12</span><span class="s2">, </span><span class="s4">5</span><span class="s2">))</span>

<span class="s3"># Plot Training &amp; Validation Loss</span>
<span class="s1">plt</span><span class="s2">.</span><span class="s1">subplot</span><span class="s2">(</span><span class="s4">1</span><span class="s2">, </span><span class="s4">2</span><span class="s2">, </span><span class="s4">1</span><span class="s2">)</span>
<span class="s1">plt</span><span class="s2">.</span><span class="s1">plot</span><span class="s2">(</span><span class="s1">range</span><span class="s2">(</span><span class="s4">1</span><span class="s2">, </span><span class="s1">num_epochs</span><span class="s2">+</span><span class="s4">1</span><span class="s2">), </span><span class="s1">train_losses</span><span class="s2">, </span><span class="s1">label</span><span class="s2">=</span><span class="s5">'Train Loss'</span><span class="s2">, </span><span class="s1">marker</span><span class="s2">=</span><span class="s5">'o'</span><span class="s2">)</span>
<span class="s1">plt</span><span class="s2">.</span><span class="s1">plot</span><span class="s2">(</span><span class="s1">range</span><span class="s2">(</span><span class="s4">1</span><span class="s2">, </span><span class="s1">num_epochs</span><span class="s2">+</span><span class="s4">1</span><span class="s2">), </span><span class="s1">val_losses</span><span class="s2">, </span><span class="s1">label</span><span class="s2">=</span><span class="s5">'Val Loss'</span><span class="s2">, </span><span class="s1">marker</span><span class="s2">=</span><span class="s5">'o'</span><span class="s2">)</span>
<span class="s1">plt</span><span class="s2">.</span><span class="s1">xlabel</span><span class="s2">(</span><span class="s5">'Epochs'</span><span class="s2">)</span>
<span class="s1">plt</span><span class="s2">.</span><span class="s1">ylabel</span><span class="s2">(</span><span class="s5">'Loss'</span><span class="s2">)</span>
<span class="s1">plt</span><span class="s2">.</span><span class="s1">title</span><span class="s2">(</span><span class="s5">'Training &amp; Validation Loss'</span><span class="s2">)</span>
<span class="s1">plt</span><span class="s2">.</span><span class="s1">legend</span><span class="s2">()</span>

<span class="s3"># Plot Training &amp; Validation Accuracy</span>
<span class="s1">plt</span><span class="s2">.</span><span class="s1">subplot</span><span class="s2">(</span><span class="s4">1</span><span class="s2">, </span><span class="s4">2</span><span class="s2">, </span><span class="s4">2</span><span class="s2">)</span>
<span class="s1">plt</span><span class="s2">.</span><span class="s1">plot</span><span class="s2">(</span><span class="s1">range</span><span class="s2">(</span><span class="s4">1</span><span class="s2">, </span><span class="s1">num_epochs</span><span class="s2">+</span><span class="s4">1</span><span class="s2">), </span><span class="s1">train_accuracies</span><span class="s2">, </span><span class="s1">label</span><span class="s2">=</span><span class="s5">'Train Accuracy'</span><span class="s2">, </span><span class="s1">marker</span><span class="s2">=</span><span class="s5">'o'</span><span class="s2">)</span>
<span class="s1">plt</span><span class="s2">.</span><span class="s1">plot</span><span class="s2">(</span><span class="s1">range</span><span class="s2">(</span><span class="s4">1</span><span class="s2">, </span><span class="s1">num_epochs</span><span class="s2">+</span><span class="s4">1</span><span class="s2">), </span><span class="s1">val_accuracies</span><span class="s2">, </span><span class="s1">label</span><span class="s2">=</span><span class="s5">'Val Accuracy'</span><span class="s2">, </span><span class="s1">marker</span><span class="s2">=</span><span class="s5">'o'</span><span class="s2">)</span>
<span class="s1">plt</span><span class="s2">.</span><span class="s1">xlabel</span><span class="s2">(</span><span class="s5">'Epochs'</span><span class="s2">)</span>
<span class="s1">plt</span><span class="s2">.</span><span class="s1">ylabel</span><span class="s2">(</span><span class="s5">'Accuracy (%)'</span><span class="s2">)</span>
<span class="s1">plt</span><span class="s2">.</span><span class="s1">title</span><span class="s2">(</span><span class="s5">'Training &amp; Validation Accuracy'</span><span class="s2">)</span>
<span class="s1">plt</span><span class="s2">.</span><span class="s1">legend</span><span class="s2">()</span>

<span class="s1">plt</span><span class="s2">.</span><span class="s1">show</span><span class="s2">()</span>

<span class="s1">print</span><span class="s2">(</span><span class="s5">&quot;Model Training Complete!&quot;</span><span class="s2">)</span>
</pre>
</body>
</html>