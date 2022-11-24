pour prédire le contexte lié à une question, il faut exécuter le script predict_question.py avec en argument le dataset (--dataset "train_set" pour le train set de SQUAD et --dataset "val_set" pour le validation test de SQUAD). rajouter par la suite la question (--question "question_text")

forme générale de la commande:

python predict_question.py --dataset "train_set" --question "question_text"

exemple:

python predict_question.py --dataset "train_set" --question "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?"


Une fois la commande lancé il faut attendre a peut près 8 secondes avant d'obtenir la réponse ce temps contient:
- le temps de chargement de données neccessaires à la création du model
- le temps de création du model
- le temps de prédiction de la question. c'est ce temps qui est affiché à la suite du context prédit. il est inférieur à 1s