<h1>To Do</h1>
<ul>
    <li>Proper Noun should be ignored during synonyms generations by ChatGPT like person name, brand name etc.</li>
    <li>Sequnce numbers like card number, pincode should be ignored during synonyms generation.</li>
</ul>

<h1>Report</h1>
<p>
<h2>Home Decor</h2>
    <pre>
Classification / Extraction
---------------------------
                precision    recall  f1-score   support

Classification       1.00      1.00      1.00        20

      accuracy                           1.00        20
     macro avg       1.00      1.00      1.00        20
  weighted avg       1.00      1.00      1.00        20
    </pre>
    <pre>
Single Value / Multi Value
---------------------------
              precision    recall  f1-score   support

 Multi Value       0.00      0.00      0.00         0
Single Value       1.00      0.35      0.52        20

    accuracy                           0.35        20
   macro avg       0.50      0.17      0.26        20
weighted avg       1.00      0.35      0.52        20
    </pre>
    <pre>
Input Priority
--------------
              precision    recall  f1-score   support

 Image, Text       0.00      0.00      0.00         0
  Image,Text       0.00      0.00      0.00         0
        Text       0.73      0.80      0.76        10
 Text, Image       0.67      0.40      0.50        10

    accuracy                           0.60        20
   macro avg       0.35      0.30      0.32        20
weighted avg       0.70      0.60      0.63        20
    </pre>
</p>

<p>
<h2>Fashion</h2>
<pre>
Classification / Extraction
---------------------------
                 precision    recall  f1-score   support

 Classification       0.72      1.00      0.84        36
Classification        0.00      0.00      0.00         1
     Extraction       0.00      0.00      0.00        13

       accuracy                           0.72        50
      macro avg       0.24      0.33      0.28        50
   weighted avg       0.52      0.72      0.60        50
</pre>
<pre>
Single Value / Multi Value
---------------------------
              precision    recall  f1-score   support

 Multi Value       0.57      0.88      0.70        26
Multi Value        0.00      0.00      0.00         2
Single Value       0.70      0.32      0.44        22

    accuracy                           0.60        50
   macro avg       0.42      0.40      0.38        50
weighted avg       0.61      0.60      0.55        50
</pre>
<pre>
Input Priority
--------------
                  precision    recall  f1-score   support

           Image       1.00      0.13      0.23        38
     Image, Text       0.00      0.00      0.00         0
      Image,Text       0.00      0.00      0.00         3
            Text       0.14      1.00      0.24         5
     Text, Image       0.00      0.00      0.00         0
Text, Image(OCR)       0.00      0.00      0.00         0
 Text,Image(OCR)       0.00      0.00      0.00         4

        accuracy                           0.20        50
       macro avg       0.16      0.16      0.07        50
    weighted avg       0.77      0.20      0.20        50
</pre>
</p>

<p>
    <h2>Beauty</h2>
    <pre>
Classification / Extraction
---------------------------

    </pre>
    <pre>
Single Value / Multi Value
---------------------------
    </pre>
    <pre>
Input Priority
--------------
    </pre>
</p>