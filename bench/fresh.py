"""Authored regression suite; previously consulted, not an untouched audit.

New claims, new domains, new sentence families: 8 claims x 10 minimal pairs (EN/PT) plus 20 picks.
Run from the repository root:
    python -m bench.fresh --output REPORT SRC [SRC ...]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from semantic_switch import Model
from bench.outcomes import destination, output_argument, report, runtime

FRESH = {
    "meeting": ("The writer accepts the proposed meeting time.", (), [
        ("Thursday at 3pm works for me, see you then.", "Thursday at 3pm doesn't work for me, can we try Friday?",
         "plain"),
        ("I can't say no to 10am, so 10am it is.", "I can't say yes to 10am; I have a dentist appointment.",
         "double_negation"),
        ("My assistant thought I was busy, but I'm free at noon. Let's do it.",
         "I thought I was free at noon, but my assistant says I'm booked. Sorry.", "role"),
        ("I first said no to Monday, but I moved things around: Monday is fine.",
         "I first said Monday was fine, but something came up, so no.", "temporal"),
        ("If nothing changes, I'll be there at 9. Nothing will change.",
         "If my flight lands early I might make 9, but don't count on it.", "hypothetical"),
        ("Quinta às 15h está ótimo para mim, nos vemos lá.", "Quinta às 15h não dá para mim, pode ser sexta?", "plain"),
        ("Não tenho como recusar o horário das 10h; confirmado.", "Não tenho como aceitar o horário das 10h; tenho médico.",
         "double_negation"),
        ("Minha secretária achou que eu estava ocupado, mas estou livre ao meio-dia. Fechado.",
         "Achei que estava livre ao meio-dia, mas minha secretária diz que não. Desculpe.", "role"),
        ("Primeiro recusei segunda, mas reorganizei a agenda: segunda está bom.",
         "Primeiro aceitei segunda, mas surgiu um imprevisto, então não.", "temporal"),
        ("O horário proposto está perfeito.", "O horário proposto é impossível para mim.", "negation"),
    ]),
    "leave": ("The employee is asking for time off.", (), [
        ("Could I take next Friday off for my brother's wedding?",
         "Could I work from home next Friday? I'll be online all day.", "contrast"),
        ("I'm not asking for a raise, I'm asking for two days of vacation.",
         "I'm not asking for vacation, I'm asking about my raise.", "contrast"),
        ("My manager suggested I keep working, but I need a week off to recover.",
         "My manager suggested I take a week off, but I'd rather keep working.", "role"),
        ("Please approve my leave from the 3rd to the 7th.", "Please cancel my leave from the 3rd to the 7th; I'll be in.",
         "negation"),
        ("I'd like to use my remaining holiday days in August.", "How many holiday days do I have left? Just checking.",
         "keyword"),
        ("Posso tirar folga na próxima sexta para o casamento do meu irmão?",
         "Posso trabalhar de casa na próxima sexta? Fico online o dia todo.", "contrast"),
        ("Não peço aumento, peço dois dias de férias.", "Não peço férias, peço informações sobre o aumento.", "contrast"),
        ("Meu gestor sugeriu que eu continuasse, mas preciso de uma semana de licença.",
         "Meu gestor sugeriu uma semana de licença, mas prefiro continuar trabalhando.", "role"),
        ("Por favor, aprovem minhas férias do dia 3 ao 7.", "Por favor, cancelem minhas férias do dia 3 ao 7; vou trabalhar.",
         "negation"),
        ("Quero usar os dias de férias que sobraram em agosto.", "Quantos dias de férias ainda tenho? Só curiosidade.",
         "keyword"),
    ]),
    "incident": ("The sender reports a security incident.", (), [
        ("Someone logged into my account from another country last night.",
         "I logged into my account from another country last night while travelling.", "role"),
        ("I clicked a link in a fake invoice email and typed my password.",
         "I got a fake invoice email and deleted it without clicking anything.", "negation"),
        ("Our server is sending spam we never wrote; I think it was hacked.",
         "Our server is slow today; probably the backup job.", "implicit"),
        ("The phishing test I failed last month was real this time: my credentials leaked.",
         "Last month's phishing scare turned out to be a drill; nothing leaked.", "temporal"),
        ("What should I do? A stranger has our VPN keys.", "What does VPN stand for? Just curious.", "keyword"),
        ("Alguém entrou na minha conta de outro país ontem à noite.",
         "Entrei na minha conta de outro país ontem à noite, estava viajando.", "role"),
        ("Cliquei no link de um boleto falso e digitei minha senha.",
         "Recebi um boleto falso e apaguei sem clicar em nada.", "negation"),
        ("Nosso servidor está enviando spam que não escrevemos; acho que foi invadido.",
         "Nosso servidor está lento hoje; deve ser o backup.", "implicit"),
        ("O susto do mês passado era treino, mas agora vazaram minhas credenciais de verdade.",
         "O susto do mês passado era só um treino; nada vazou.", "temporal"),
        ("Um desconhecido tem as chaves da nossa VPN.", "O que significa VPN? Só curiosidade.", "keyword"),
    ]),
    "heating": ("The tenant says the heating is not working.", (), [
        ("The radiators have been cold for three days.", "The radiators were cold, but the plumber fixed them yesterday.",
         "temporal"),
        ("It's not the hot water, it's the heating that's dead.", "It's not the heating, it's the hot water that's dead.",
         "contrast"),
        ("My neighbour's heating works; mine doesn't.", "My heating works; it's my neighbour's that doesn't.", "role"),
        ("When I turn the thermostat up, nothing happens.", "When I turn the thermostat up, will the bill go up too?",
         "hypothetical"),
        ("Freezing in here, the boiler won't start.", "Freezing outside today, but the flat is warm.", "implicit"),
        ("Os aquecedores estão frios há três dias.", "Os aquecedores estavam frios, mas o encanador consertou ontem.",
         "temporal"),
        ("Não é a água quente, é o aquecimento que parou.", "Não é o aquecimento, é a água quente que parou.", "contrast"),
        ("O aquecimento do vizinho funciona; o meu não.", "O meu aquecimento funciona; o do vizinho é que não.", "role"),
        ("Quando aumento o termostato, nada acontece.", "Quando aumento o termostato, a conta sobe também?",
         "hypothetical"),
        ("Está congelando aqui dentro, a caldeira não liga.", "Está congelando lá fora, mas o apartamento está quente.",
         "implicit"),
    ]),
    "pain": ("The patient says the pain has gotten worse.", (), [
        ("The pain is much stronger than yesterday.", "The pain is much weaker than yesterday.", "plain"),
        ("It's not better, it's worse since I started the new pills.",
         "It's not worse, it's better since I started the new pills.", "contrast"),
        ("My wife thinks I'm improving, but the pain keeps growing.",
         "I thought it was getting worse, but my wife says I'm improving, and she's right.", "role"),
        ("Yesterday it was mild; today I can barely walk.", "Yesterday I could barely walk; today it's mild.", "temporal"),
        ("I can't say the pain has eased; it's worse.", "I can't say the pain has grown; it's easing.", "double_negation"),
        ("A dor está muito mais forte que ontem.", "A dor está muito mais fraca que ontem.", "plain"),
        ("Não melhorou, piorou desde o remédio novo.", "Não piorou, melhorou desde o remédio novo.", "contrast"),
        ("Minha esposa acha que estou melhorando, mas a dor só aumenta.",
         "Eu achava que piorava, mas minha esposa diz que melhorei, e ela tem razão.", "role"),
        ("Ontem era leve; hoje mal consigo andar.", "Ontem mal conseguia andar; hoje está leve.", "temporal"),
        ("Não posso dizer que a dor diminuiu; aumentou.", "Não posso dizer que a dor aumentou; está diminuindo.",
         "double_negation"),
    ]),
    "shipped": ("The supplier confirms the shipment has left the warehouse.", (), [
        ("Your pallets left our warehouse this morning; tracking attached.",
         "Your pallets are packed and will leave our warehouse tomorrow.", "temporal"),
        ("We didn't hold it back: it shipped on Monday.", "We didn't ship it: it's on hold until payment clears.",
         "negation"),
        ("The courier collected the boxes an hour ago.", "The courier is scheduled to collect the boxes at 5pm.",
         "temporal"),
        ("If you're asking whether it shipped: yes, it's on the truck.",
         "If it ships today you'll get tracking; it hasn't yet.", "hypothetical"),
        ("Our warehouse released order 881 to the carrier.", "Our warehouse is still picking order 881.", "implicit"),
        ("Seus paletes saíram do nosso armazém hoje cedo; segue o rastreio.",
         "Seus paletes estão embalados e saem do armazém amanhã.", "temporal"),
        ("Não seguramos nada: foi despachado na segunda.", "Não despachamos: está retido até o pagamento compensar.",
         "negation"),
        ("A transportadora retirou as caixas há uma hora.", "A transportadora vai retirar as caixas às 17h.", "temporal"),
        ("Se a pergunta é se foi enviado: sim, já está no caminhão.",
         "Se for enviado hoje, mandamos o rastreio; ainda não foi.", "hypothetical"),
        ("O armazém liberou o pedido 881 para a transportadora.", "O armazém ainda está separando o pedido 881.",
         "implicit"),
    ]),
    "decline": ("The writer declines the job offer.", (), [
        ("Thank you, but I've decided not to accept the position.",
         "Thank you, I've decided to accept the position.", "negation"),
        ("I'm not declining because of the salary; I'm declining because of the commute.",
         "I'm not accepting because of the salary; I'm accepting because of the team.", "contrast"),
        ("My family wanted me to take it, but I'm turning it down.",
         "My family wanted me to turn it down, but I'm taking it.", "role"),
        ("I was going to accept, but I've changed my mind and must say no.",
         "I was going to say no, but I've changed my mind and accept.", "temporal"),
        ("I'm afraid I'll have to pass on this one.", "Could we discuss the start date? I'm very keen.", "implicit"),
        ("Obrigado, mas decidi não aceitar a vaga.", "Obrigado, decidi aceitar a vaga.", "negation"),
        ("Não recuso pelo salário; recuso pela distância.", "Não aceito pelo salário; aceito pela equipe.", "contrast"),
        ("Minha família queria que eu aceitasse, mas vou recusar.",
         "Minha família queria que eu recusasse, mas vou aceitar.", "role"),
        ("Ia aceitar, mas mudei de ideia e preciso dizer não.", "Ia recusar, mas mudei de ideia e aceito.", "temporal"),
        ("Infelizmente vou ter que passar essa.", "Podemos conversar sobre a data de início? Estou animado.", "implicit"),
    ]),
    "address": ("The customer wants to change the delivery address.", (), [
        ("Please send order 412 to my office instead of my home.", "Please confirm order 412 is going to my home address.",
         "plain"),
        ("Not to the old flat — I moved; deliver to Rua Augusta 10.",
         "Deliver to the old flat as planned; I haven't moved yet.", "negation"),
        ("My husband entered his address by mistake; it should come to mine.",
         "I entered my husband's address on purpose; keep it that way.", "role"),
        ("Can I still redirect the parcel to my parents' house?", "Can I still add a gift note to the parcel?", "keyword"),
        ("I'll be away, so ship it to my neighbour at number 12.", "I'll be away, so please hold the parcel until Monday.",
         "implicit"),
        ("Por favor, enviem o pedido 412 para meu escritório em vez da minha casa.",
         "Por favor, confirmem que o pedido 412 vai para minha casa.", "plain"),
        ("Não para o apartamento antigo — mudei; entreguem na Rua Augusta 10.",
         "Entreguem no apartamento antigo, como combinado; ainda não mudei.", "negation"),
        ("Meu marido colocou o endereço dele por engano; deve vir para o meu.",
         "Coloquei o endereço do meu marido de propósito; mantenham assim.", "role"),
        ("Ainda dá para redirecionar o pacote para a casa dos meus pais?",
         "Ainda dá para incluir um cartão de presente no pacote?", "keyword"),
        ("Vou viajar, então mandem para o vizinho do número 12.", "Vou viajar, então segurem o pacote até segunda.",
         "implicit"),
    ]),
}

FRESH_TOPICS = {
    "leave": "vacation, sick leave or time off",
    "payroll": "salary, payslips, bonuses or tax forms",
    "equipment": "laptops, phones or other work equipment",
    "access": "accounts, passwords, badges or system access",
    "benefits": "health insurance, pension or other benefits",
}
FRESH_QUESTION = "What is the employee's message mainly about?"
FRESH_PICKS = [
    ("I need Friday off for a family thing.", "leave"),
    ("My payslip shows the wrong overtime hours.", "payroll"),
    ("The laptop screen cracked when I dropped it.", "equipment"),
    ("My badge stopped opening the main door.", "access"),
    ("Does our health plan cover physiotherapy?", "benefits"),
    ("I was off sick yesterday; do I need a doctor's note?", "leave"),
    ("When will the annual bonus be paid?", "payroll"),
    ("Can I get a second monitor for my desk?", "equipment"),
    ("I'm locked out of the HR portal after a password reset.", "access"),
    ("How do I add my newborn to the dental insurance?", "benefits"),
    ("Preciso de folga na sexta para um compromisso de família.", "leave"),
    ("Meu holerite veio com as horas extras erradas.", "payroll"),
    ("A tela do notebook trincou quando caiu.", "equipment"),
    ("Meu crachá parou de abrir a porta principal.", "access"),
    ("O plano de saúde cobre fisioterapia?", "benefits"),
    ("Fiquei doente ontem; preciso de atestado?", "leave"),
    ("Quando sai o bônus anual?", "payroll"),
    ("Posso pedir um segundo monitor?", "equipment"),
    ("Fiquei sem acesso ao portal de RH depois de trocar a senha.", "access"),
    ("Como incluo meu filho recém-nascido no plano odontológico?", "benefits"),
]


def main():
    from bench.model_comparison import suite
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+")
    output_argument(parser)
    args = parser.parse_args()
    destination(args.output)
    out = {}
    for source in args.sources:
        with runtime(Model(source)) as rt:
            result = suite(rt, checks_spec=FRESH, picks_spec=FRESH_PICKS, topics=FRESH_TOPICS, question=FRESH_QUESTION)
        out[Path(source.split(":")[-1]).stem] = result
        keys = ("accuracy_given_prediction", "correct_per_submission", "auc", "coverage_0.9", "precision_0.9", "pick_accuracy")
        print(Path(source.split(":")[-1]).stem, {k: result["summary"][k] for k in keys}, flush=True)
    report(args.output, out)


if __name__ == "__main__":
    main()
