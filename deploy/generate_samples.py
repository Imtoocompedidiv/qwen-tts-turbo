"""Generate audio samples via the running server. Fast (no model reload)."""
import os
import sys
import urllib.request
import urllib.parse

SAMPLES = [
    ("Bonjour et bienvenue. Merci de nous avoir contactes. Je suis Vivian et je suis la pour vous aider. Comment puis-je vous etre utile aujourd'hui ? N'hesitez pas a me poser toutes vos questions, je ferai de mon mieux pour y repondre.", "Vivian", "French", "", "fr_vivian"),
    ("Je comprends tout a fait votre situation et je vous assure que nous allons trouver une solution ensemble. Laissez-moi verifier votre dossier. D'apres les informations dont je dispose, votre remboursement a bien ete initie le quinze mars dernier. Le montant devrait apparaitre sur votre releve dans cinq a sept jours ouvres.", "Serena", "French", "Voix douce et rassurante", "fr_serena_doux"),
    ("Good morning, thank you for calling our customer service center. My name is Dylan and I will be assisting you today. I can see your account right here. It looks like your subscription was renewed last week, and I can confirm that everything is in order. Is there anything else I can help you with?", "Dylan", "English", "", "en_dylan"),
    ("Great news! I have just checked with our logistics team and your package is currently on its way. It was dispatched from our warehouse yesterday morning and according to the tracking information, it should arrive at your doorstep within the next two business days. We really appreciate your patience!", "Eric", "English", "Ton dynamique et enthousiaste", "en_eric_dynamic"),
    ("Guten Tag und herzlich willkommen bei unserem Kundenservice. Mein Name ist Vivian und ich bin heute fur Sie da. Ich habe Ihre Anfrage erhalten und werde mich sofort darum kummern. Bitte haben Sie einen Moment Geduld wahrend ich Ihre Daten uberprufe.", "Vivian", "German", "", "de_vivian"),
    ("Buenos dias, muchas gracias por comunicarse con nosotros. Mi nombre es Vivian y estoy aqui para ayudarle. He revisado su expediente y puedo confirmar que su solicitud ha sido procesada correctamente. El reembolso se efectuara en un plazo de cinco a siete dias habiles.", "Vivian", "Spanish", "", "es_vivian"),
    ("Buongiorno e benvenuto. Sono Vivian e saro lieta di assisterla oggi. Ho verificato il suo dossier e posso confermare che la sua richiesta e stata elaborata correttamente. Il rimborso verra effettuato entro cinque-sette giorni lavorativi.", "Vivian", "Italian", "", "it_vivian"),
    ("Bom dia e obrigado por entrar em contato conosco. Meu nome e Vivian e estou aqui para ajuda-lo. Verifiquei o seu processo e posso confirmar que a sua solicitacao foi processada corretamente. O reembolso sera efetuado dentro de cinco a sete dias uteis.", "Vivian", "Portuguese", "", "pt_vivian"),
    ("Bonjour, je vous remercie pour votre appel. Permettez-moi de vous informer que votre commande a ete expediee ce matin depuis notre entrepot. Selon les informations de suivi, la livraison est prevue pour vendredi prochain entre neuf heures et dix-sept heures.", "Ryan", "French", "", "fr_ryan"),
    ("Welcome to our support line. I understand this situation has been frustrating for you, and I sincerely apologize for the inconvenience. Let me look into this right away. I can see that there was a billing error on your last invoice. I am going to correct this immediately and issue a credit to your account.", "Aiden", "English", "", "en_aiden"),
]

def main():
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "samples"
    os.makedirs(out_dir, exist_ok=True)

    for text, voice, lang, instruct, name in SAMPLES:
        params = urllib.parse.urlencode({"text": text, "voice": voice, "language": lang, "instruct": instruct})
        url = f"{base_url}/generate?{params}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            resp = urllib.request.urlopen(req, timeout=120)
            data = resp.read()
            path = os.path.join(out_dir, f"{name}.wav")
            with open(path, "wb") as f:
                f.write(data)
            print(f"  {name}.wav: {len(data)//1024}KB")
        except Exception as e:
            print(f"  {name}: FAILED - {e}")

if __name__ == "__main__":
    main()
