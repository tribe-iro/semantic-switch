from __future__ import annotations

import time
import json

from bench.outcomes import materialize
from collections import Counter

import numpy as np

from semantic_switch import files
from semantic_switch.judgment import DEFAULT, Judgment, Policy, outcome

SUITE_VERSION = "2"
LEGAL_EXCLUDES = ("merely mentioning a lawyer", "quoting someone else's threat")

CHECKS = {
    "refund": ("The customer asks to receive money back for a charge.", (), [
        ("I was charged twice for May. Please refund one of the charges.",
         "I was charged twice for May, but the second charge was already refunded. Thanks.", "temporal"),
        ("Don't send a replacement; I want my money back.",
         "Don't give me my money back; I want a replacement.", "contrast"),
        ("My friend told me not to bother asking, but I am asking: refund my last payment.",
         "My friend told me to ask for a refund, but I don't want one. Just fix the invoice address.", "reported"),
        ("Reverse the $40 charge from yesterday, it was not authorized.",
         "The $40 charge from yesterday is correct, I just need a copy of the receipt.", "implicit"),
        ("If you can't fix it today, refund me. And you clearly can't, so refund me.",
         "If the refund policy covers annual plans, I may ask for one next year.", "hypothetical"),
        ("I'm not asking for a discount, I'm asking for the payment back.",
         "I'm not asking for the payment back, I'm asking for a discount on the next one.", "negation"),
        ("Fui cobrado duas vezes este mês. Quero o estorno de uma das cobranças.",
         "Fui cobrado duas vezes este mês, mas vocês já estornaram uma. Obrigado.", "temporal"),
        ("Não quero troca, quero meu dinheiro de volta.",
         "Não quero meu dinheiro de volta, quero a troca.", "contrast"),
        ("Devolvam o valor que paguei pelo plano anual.",
         "Quanto custa o plano anual? Ainda não paguei nada.", "keyword"),
        ("Minha esposa disse para não pedir reembolso, mas eu peço: devolvam o pagamento.",
         "Minha esposa disse para pedir reembolso, mas eu não quero. Só corrijam o endereço.", "reported"),
        ("A cobrança de R$ 89 não foi autorizada; cancelem e devolvam o valor.",
         "A cobrança de R$ 89 está correta; preciso apenas da nota fiscal.", "implicit"),
        ("Não é que eu não queira o reembolso: eu quero, sim.",
         "Não é que eu queira o reembolso: só quero entender a cobrança.", "double_negation"),
    ]),
    "legal": ("The customer threatens legal action against us.", LEGAL_EXCLUDES, [
        ("Fix this by Friday or my lawyer will file a lawsuit.",
         "My lawyer asked me to request an invoice for my records.", "exception"),
        ("I will take you to small claims court if this is not resolved.",
         "I work at a small claims court, so I know support is hard. Just a question about my plan.", "keyword"),
        ("This is my last warning before I sue you.",
         "My neighbour said he would sue you, but I just want my password reset.", "reported"),
        ("I am filing a complaint with the consumer agency and a lawsuit against your company.",
         "I'm a lawyer myself, and I'd like to upgrade my plan.", "exception"),
        ("I'm not bluffing: my attorney is preparing the case against you.",
         "I'm not suing anyone, I just want the bug fixed.", "negation"),
        ("Keep ignoring me and you'll hear from my lawyers.",
         "Ignore my earlier message about lawyers; I calmed down and won't take any action.", "temporal"),
        ("Se não resolverem até sexta, vou entrar com um processo.",
         "Meu advogado pediu a segunda via do boleto.", "exception"),
        ("Vou acionar vocês no Procon e na justiça.",
         "Trabalho no Procon e queria tirar uma dúvida sobre meu plano.", "keyword"),
        ("Esta é a última vez que peço antes de processar a empresa.",
         "Meu vizinho disse que vai processar vocês, mas eu só quero trocar a senha.", "reported"),
        ("Meu advogado já está preparando a ação contra vocês.",
         "Não vou processar ninguém, só quero que corrijam o erro.", "negation"),
        ("Continuem me ignorando e vocês vão se ver com meu advogado.",
         "Esqueçam o que falei sobre advogado; já me acalmei e não vou fazer nada.", "temporal"),
        ("Não pensem que é blefe: vou à justiça.",
         "Não é que eu vá à justiça; só estou decepcionado.", "double_negation"),
    ]),
    "cancel": ("The customer wants to cancel their subscription.", (), [
        ("Please cancel my subscription at the end of this month.",
         "Please don't cancel my subscription; the payment failure was my bank's fault.", "negation"),
        ("I've decided to stop my plan. Turn off the renewal.",
         "I've decided to keep my plan, even though I thought about cancelling last week.", "temporal"),
        ("My manager told me to keep the subscription, but I want it cancelled.",
         "My manager told me to cancel the subscription, but I'm keeping it.", "role"),
        ("How do I cancel? I don't want to be charged again.",
         "How do I cancel an order for a single item? I love my subscription.", "keyword"),
        ("Not upgrading, not downgrading: ending it.",
         "Not ending it, just downgrading to the basic tier.", "contrast"),
        ("I'm moving abroad and won't need the service anymore, so stop the subscription.",
         "I'm moving abroad; can I keep using my subscription there?", "implicit"),
        ("Quero cancelar minha assinatura a partir do mês que vem.",
         "Não cancelem minha assinatura; o pagamento falhou por erro do banco.", "negation"),
        ("Decidi encerrar o plano. Não renovem mais.",
         "Pensei em cancelar semana passada, mas decidi continuar com o plano.", "temporal"),
        ("Como faço para cancelar? Não quero mais ser cobrado.",
         "Como cancelo um pedido avulso? Adoro minha assinatura.", "keyword"),
        ("Meu sócio quer manter a assinatura, mas eu quero cancelar.",
         "Meu sócio quer cancelar a assinatura, mas eu vou manter.", "role"),
        ("Nem upgrade nem downgrade: quero encerrar.",
         "Não quero encerrar, só mudar para o plano básico.", "contrast"),
        ("Vou me mudar e não vou mais usar o serviço; parem a assinatura.",
         "Vou me mudar; consigo usar a assinatura no novo endereço?", "implicit"),
    ]),
    "human": ("The customer asks to talk to a human agent.", (), [
        ("Stop the bot answers. I need to speak with a real person.",
         "Your bot answered perfectly, no need for a person. Thanks!", "negation"),
        ("Can someone from your team call me?",
         "Can your app notify me automatically when the order ships?", "implicit"),
        ("Transfer me to an agent, please.",
         "Last week the agent I was transferred to solved it; today I only need the FAQ link.", "temporal"),
        ("I don't want to chat with a machine, put me through to a human.",
         "I don't need a human, the machine can handle a password reset.", "contrast"),
        ("My son keeps saying the bot is fine; I still want a person.",
         "My son wants a person on the line; I'm fine with the bot.", "role"),
        ("Is there any human there? Please, a human.",
         "Is there a human-readable version of the invoice?", "keyword"),
        ("Quero falar com um atendente de verdade, não com robô.",
         "O robô resolveu meu problema, não preciso de atendente.", "negation"),
        ("Alguém da equipe pode me ligar?",
         "O aplicativo pode me avisar quando o pedido sair?", "implicit"),
        ("Me transfere para um humano, por favor.",
         "Semana passada um atendente resolveu; hoje só quero o link do FAQ.", "temporal"),
        ("Meu filho acha que o robô basta; eu quero uma pessoa.",
         "Meu filho quer falar com uma pessoa; eu estou satisfeito com o robô.", "role"),
        ("Não quero conversar com máquina, quero um atendente.",
         "Não preciso de atendente, a máquina resolve a troca de senha.", "contrast"),
        ("Tem alguém aí? Uma pessoa, por favor.",
         "Tem alguma pessoa da família que possa usar meu plano?", "keyword"),
    ]),
    "broken": ("The customer reports that the product is not working.", (), [
        ("The app crashes every time I open it.",
         "The app used to crash, but since the last update it works perfectly.", "temporal"),
        ("Nothing loads on the dashboard since this morning.",
         "The dashboard loads fine; I just want to change the color theme.", "negation"),
        ("I'm not saying it's slow, I'm saying it doesn't start at all.",
         "I'm not saying it doesn't work, I'm saying the font is small.", "contrast"),
        ("My colleague says the export works for him, but on my account it fails with an error.",
         "My colleague thought the export was broken, but it was his browser; it works for both of us.", "role"),
        ("Error 500 when I try to pay.",
         "What does error 500 mean in general? Just curious, everything works.", "keyword"),
        ("When I click save, my changes disappear.",
         "When I click save, will my changes sync to my phone too?", "hypothetical"),
        ("O aplicativo fecha sozinho toda vez que abro.",
         "O aplicativo fechava sozinho, mas depois da atualização funciona bem.", "temporal"),
        ("Não carrega nada no painel desde cedo.",
         "O painel carrega normal; só quero mudar o tema.", "negation"),
        ("Dá erro 500 quando tento pagar.",
         "O que significa erro 500? Só curiosidade, está tudo funcionando.", "keyword"),
        ("Quando clico em salvar, as alterações somem.",
         "Quando clico em salvar, as alterações vão para o celular também?", "hypothetical"),
        ("Não estou dizendo que está lento; não abre de jeito nenhum.",
         "Não estou dizendo que não funciona; só acho a letra pequena.", "contrast"),
        ("Não é verdade que esteja funcionando: continua travando.",
         "Não é verdade que esteja travando: funciona bem, foi engano meu.", "double_negation"),
    ]),
    "late": ("The customer complains that an order has not arrived.", (), [
        ("It's been three weeks and my package still hasn't arrived.",
         "My package arrived today, three weeks late, but it's here.", "temporal"),
        ("The tracking says delivered, but there is nothing at my door.",
         "The tracking says delivered and I have the box. Can I return one item?", "implicit"),
        ("I'm not complaining about the quality; the order never came.",
         "I'm not complaining about the delivery; the quality is poor.", "contrast"),
        ("My sister's order arrived, mine didn't.",
         "My order arrived; my sister's didn't, and she will contact you herself.", "role"),
        ("Where is my order #4512? Still waiting.",
         "Where can I see my order history? Just for my records.", "keyword"),
        ("It was supposed to arrive Monday; it's Thursday and nothing.",
         "It is supposed to arrive Monday; can I change the delivery address before then?", "hypothetical"),
        ("Já faz três semanas e meu pedido não chegou.",
         "Meu pedido chegou hoje, atrasado, mas chegou.", "temporal"),
        ("O rastreio diz entregue, mas não recebi nada.",
         "O rastreio diz entregue e estou com a caixa. Posso trocar um item?", "implicit"),
        ("Cadê meu pedido 4512? Continuo esperando.",
         "Onde vejo meu histórico de pedidos? Só para controle.", "keyword"),
        ("O pedido da minha irmã chegou, o meu não.",
         "O meu chegou; o da minha irmã é que não, e ela mesma vai falar com vocês.", "role"),
        ("Não reclamo da qualidade; o pedido nunca veio.",
         "Não reclamo da entrega; a qualidade é que é ruim.", "contrast"),
        ("Era para chegar segunda; já é quinta e nada.",
         "Deve chegar segunda; posso mudar o endereço antes?", "hypothetical"),
    ]),
    "solved": ("The customer says their problem is now solved.", (), [
        ("Never mind, it works now. Thanks!",
         "It worked for a minute, but now it's broken again.", "temporal"),
        ("I restarted the router as you said and everything is fine now.",
         "I restarted the router as you said and nothing changed.", "plain"),
        ("You can close the ticket, all good.",
         "Please don't close the ticket, it's not fixed.", "negation"),
        ("My husband thought it was still broken, but I checked: it's fixed.",
         "I thought it was fixed, but my husband checked: it's still broken.", "role"),
        ("No need to reply anymore, I found the setting myself.",
         "No reply yet, and I still can't find the setting.", "keyword"),
        ("It's not that it's still failing; it's fine now.",
         "It's not that it's fine now; it's still failing.", "double_negation"),
        ("Deixa pra lá, já funcionou. Obrigado!",
         "Funcionou por um minuto, mas voltou a dar erro.", "temporal"),
        ("Reiniciei o roteador como pediram e agora está tudo certo.",
         "Reiniciei o roteador como pediram e nada mudou.", "plain"),
        ("Podem fechar o chamado, está tudo bem.",
         "Não fechem o chamado, ainda não resolveu.", "negation"),
        ("Meu marido achava que ainda estava com defeito, mas conferi: está resolvido.",
         "Eu achava que estava resolvido, mas meu marido conferiu: continua com defeito.", "role"),
        ("Não precisa mais responder, achei a configuração sozinho.",
         "Ninguém respondeu ainda e continuo sem achar a configuração.", "keyword"),
        ("Não é que continue falhando; agora está ok.",
         "Não é que agora esteja ok; continua falhando.", "double_negation"),
    ]),
    "erase": ("The customer asks us to delete their personal data.", (), [
        ("Under GDPR, delete all the personal data you hold about me.",
         "Under GDPR, send me a copy of the personal data you hold about me.", "contrast"),
        ("Erase my account and everything you stored about me.",
         "Don't erase my account, I only want to change my email.", "negation"),
        ("Remove my phone number and address from your systems.",
         "Add my new phone number and address to your systems.", "plain"),
        ("I asked last month to keep my data; I've changed my mind, delete it.",
         "I asked last month to delete my data; I've changed my mind, keep it.", "temporal"),
        ("Please wipe the photos and records you keep about me.",
         "Please back up the photos I uploaded so I don't lose them.", "keyword"),
        ("My lawyer says I can't force you to delete my data, but I'm asking you to anyway.",
         "My lawyer says I can force you to delete my data, but I don't want that; just stop the emails.",
         "reported"),
        ("Pela LGPD, apaguem todos os meus dados pessoais.",
         "Pela LGPD, me enviem uma cópia dos meus dados pessoais.", "contrast"),
        ("Excluam minha conta e tudo o que guardaram sobre mim.",
         "Não excluam minha conta, só quero trocar o e-mail.", "negation"),
        ("Removam meu telefone e endereço do sistema de vocês.",
         "Adicionem meu novo telefone e endereço ao sistema.", "plain"),
        ("Pedi mês passado para manter meus dados; mudei de ideia, apaguem.",
         "Pedi mês passado para apagar meus dados; mudei de ideia, mantenham.", "temporal"),
        ("Apaguem as fotos e registros que vocês têm sobre mim.",
         "Façam backup das fotos que enviei para eu não perder.", "keyword"),
        ("Meu advogado diz que não posso obrigar vocês a apagar meus dados, mas peço mesmo assim.",
         "Meu advogado diz que posso obrigar vocês a apagar meus dados, mas não quero; só parem os e-mails.",
         "reported"),
    ]),
    "price": ("The customer asks how much something costs.", (), [
        ("How much is the premium plan per month?",
         "I already pay for the premium plan; how do I add a user?", "keyword"),
        ("What would it cost to add two more seats?",
         "I added two more seats; the cost was fine.", "temporal"),
        ("Before I buy, tell me the price of the 1TB option.",
         "I bought the 1TB option; tell me how to activate it.", "plain"),
        ("Is the yearly plan cheaper? What's the price?",
         "The yearly plan is cheaper, I know. I want to switch to it.", "implicit"),
        ("My boss knows the price, I don't; how much is it?",
         "I know the price; my boss doesn't, and he'll check your website.", "role"),
        ("I'm not asking about features, just the price.",
         "I'm not asking about the price, just the features.", "contrast"),
        ("Quanto custa o plano premium por mês?",
         "Já pago o plano premium; como adiciono um usuário?", "keyword"),
        ("Quanto sairia adicionar mais dois usuários?",
         "Adicionei mais dois usuários; o valor ficou ok.", "temporal"),
        ("Antes de comprar, me diz o preço da opção de 1TB.",
         "Comprei a opção de 1TB; como ativo?", "plain"),
        ("Meu chefe sabe o preço, eu não; quanto é?",
         "Eu sei o preço; meu chefe não, e ele vai olhar no site.", "role"),
        ("Não pergunto das funções, só do preço.",
         "Não pergunto do preço, só das funções.", "contrast"),
        ("O plano anual é mais barato? Qual o valor?",
         "O plano anual é mais barato, eu sei. Quero mudar para ele.", "implicit"),
    ]),
    "today": ("The customer says they need a solution today.", (), [
        ("I need this fixed today, my store opens tomorrow morning.",
         "No rush at all, next week is fine.", "plain"),
        ("It can't wait until tomorrow.",
         "It can wait until tomorrow, or even later.", "negation"),
        ("My partner says it can wait; it can't, I need it today.",
         "My partner says it's urgent; it isn't, whenever you can.", "role"),
        ("Yesterday it wasn't urgent, but now I need it solved before tonight.",
         "Yesterday it was urgent, but I found a workaround, so no hurry.", "temporal"),
        ("Today, please. Today.",
         "Today I'm just collecting information for a project next quarter.", "keyword"),
        ("It's not that I can wait: I need it today.",
         "It's not that I need it today: I can wait.", "double_negation"),
        ("Preciso disso resolvido hoje, minha loja abre amanhã cedo.",
         "Sem pressa nenhuma, semana que vem está ótimo.", "plain"),
        ("Não pode esperar até amanhã.",
         "Pode esperar até amanhã, ou mais.", "negation"),
        ("Meu sócio diz que pode esperar; não pode, preciso hoje.",
         "Meu sócio diz que é urgente; não é, quando puderem.", "role"),
        ("Ontem não era urgente, mas agora preciso antes da noite.",
         "Ontem era urgente, mas achei uma alternativa, então sem pressa.", "temporal"),
        ("Hoje, por favor. Hoje.",
         "Hoje só estou levantando informações para um projeto do próximo trimestre.", "keyword"),
        ("Não é que eu possa esperar: preciso hoje.",
         "Não é que eu precise hoje: posso esperar.", "double_negation"),
    ]),
}

TOPICS = {
    "billing": "charges, invoices, payments or refunds",
    "technical": "the product is broken, erroring or slow",
    "account": "login, password, profile or personal data",
    "shipping": "delivery and order tracking",
    "sales": "prices, plans and buying",
}
TOPIC_QUESTION = "What help is primarily requested? Choose technical for product errors even during payment; account for login/profile access even when invoices are mentioned; otherwise choose the category of the requested help."
PICKS = [
    ("I was charged twice for May.", "billing"),
    ("The app crashes as soon as I open it.", "technical"),
    ("I forgot my password and can't get in.", "account"),
    ("My package was due Monday and it's still not here.", "shipping"),
    ("What does the pro plan cost per year?", "sales"),
    ("The checkout page throws error 500 when I try to pay.", "technical"),
    ("I can't log in, so I can't even see my invoices.", "account"),
    ("The refund you promised still hasn't reached my bank.", "billing"),
    ("Please update the email on my profile.", "account"),
    ("The tracking page says delivered but the box isn't here.", "shipping"),
    ("Is there a discount if I buy ten licenses?", "sales"),
    ("Everything is slow since your update; pages take a minute.", "technical"),
    ("Fui cobrado duas vezes em maio.", "billing"),
    ("O aplicativo fecha assim que abro.", "technical"),
    ("Esqueci minha senha e não consigo entrar.", "account"),
    ("Meu pedido era para segunda e ainda não chegou.", "shipping"),
    ("Quanto custa o plano pro por ano?", "sales"),
    ("A página de pagamento dá erro 500 quando tento pagar.", "technical"),
    ("Não consigo entrar, então nem vejo minhas faturas.", "account"),
    ("O estorno prometido ainda não caiu na minha conta.", "billing"),
    ("Atualizem o e-mail do meu perfil, por favor.", "account"),
    ("O rastreio diz entregue, mas a caixa não está aqui.", "shipping"),
    ("Tem desconto se eu comprar dez licenças?", "sales"),
    ("Está tudo lento desde a atualização; as páginas levam um minuto.", "technical"),
]


def check_items(checks: dict | None = None):
    for fact, (claim, excludes, pairs) in (checks or CHECKS).items():
        for k, (pos, neg, tag) in enumerate(pairs):
            lang = "en" if k < len(pairs) // 2 else "pt"
            yield {"fact": fact, "claim": claim, "excludes": excludes, "text": pos, "label": True, "tag": tag,
                   "lang": lang, "pair": f"{fact}:{k}", "id": f"{fact}:{k}:positive"}
            yield {"fact": fact, "claim": claim, "excludes": excludes, "text": neg, "label": False, "tag": tag,
                   "lang": lang, "pair": f"{fact}:{k}", "id": f"{fact}:{k}:negative"}


def render(text: str, actor: str | None = None) -> str:
    return f"[{actor}] message: {text}" if actor else text


def _role_runtime(rt, role):
    from semantic_switch import Runtime
    selected = rt if role == "judge" else Runtime({"judge": rt.judge(role)})
    class Bound:
        def capacity(self):
            return selected.batch_inputs()

        def ask_many(self, states, items):
            return selected.ask_many(states, items)

        def check(self, state, claim, **kwargs):
            return selected.check(state, claim, **kwargs)

        def pick(self, state, question, options, **kwargs):
            return selected.pick(state, question, options, **kwargs)
    return Bound()


def run(rt, role: str = "judge", checks_spec: dict | None = None, picks_spec: list | None = None,
        topics: dict | None = None, question: str = TOPIC_QUESTION) -> dict:
    from semantic_switch.compiler import digest
    fixture = digest([SUITE_VERSION, "neutral evidence", checks_spec or CHECKS, picks_spec if picks_spec is not None else PICKS, topics or TOPICS, question])
    identity = rt.identity(role)
    rt = _role_runtime(rt, role)
    rt.check(render("warm up"), "The text is about food.")
    from semantic_switch import Check, Pick
    items = list(check_items(checks_spec))
    checks, picks, groups = [None] * len(items), [], {}
    capacity = rt.capacity()
    for index, item in enumerate(items):
        groups.setdefault((item["claim"], tuple(item["excludes"])), []).append((index, item))
    for (claim, excludes), group in groups.items():
        for start in range(0, len(group), capacity):
            window = group[start:start + capacity]
            began = time.perf_counter()
            results = rt.ask_many([render(item["text"]) for _, item in window], {"check": Check(claim, excludes)})
            json_output = materialize(results)
            json.dumps(json_output, allow_nan=False)
            elapsed = (time.perf_counter() - began) * 1000
            for (index, item), result in zip(window, results):
                result = result["check"]
                checks[index] = {key: item[key] for key in ("id", "fact", "label", "tag", "lang", "pair")} | {
                    "input": item["text"], "criterion": claim, "excludes": list(excludes), "guidance": {},
                    "p": result.p_true, "scores": result.scores, "mass": result.mass, "error": result.error, "status": result.status, "value": result.value,
                    "ms": elapsed / len(window), "batch_ms": elapsed, "batch_size": len(window)}
    pick_items = PICKS if picks_spec is None else picks_spec
    for start in range(0, len(pick_items), capacity):
        window = pick_items[start:start + capacity]
        began = time.perf_counter()
        results = rt.ask_many([render(text) for text, _ in window], {"pick": Pick(question, topics or TOPICS)})
        json.dumps(materialize(results), allow_nan=False)
        elapsed = (time.perf_counter() - began) * 1000
        for (text, answer), result in zip(window, results):
            result = result["pick"]
            picks.append({"id": f"pick:{len(picks)}", "input": text, "criterion": question, "options": topics or TOPICS, "guidance": {},
                          "presentations": result.as_dict(True)["presentations"], "answer": answer, "scores": result.scores, "mass": result.mass, "error": result.error,
                          "status": result.status, "batch_size": len(window),
                          "batch_ms": elapsed, "ms": elapsed / len(window)})
    return {"fixture_id": fixture, "role": role, "predictor": identity, "population": "bundled semantic development probe; not application accuracy", "checks": checks, "picks": picks, "timing": "amortized batch service; not per-request latency"}


def _auc(p, y):
    from semantic_switch.metrics import auc
    return auc(p, y)


def _ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    conf = np.maximum(p, 1 - p)
    right = (p >= 0.5) == y
    idx = np.minimum(((conf - 0.5) * 2 * bins).astype(int), bins - 1)
    return float(sum(abs(conf[idx == b].mean() - right[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def decision(row: dict, policy: Policy = DEFAULT) -> tuple[str, bool | None]:
    if policy == DEFAULT and "value" in row:
        return row["status"], row["value"]
    if row["p"] is None:
        return "failed", None
    status, index, _ = outcome(Judgment.check(row["criterion"], row["excludes"]), tuple(row["scores"].values()) if "scores" in row else (row["p"], 1 - row["p"]), policy, row["mass"])
    return status, None if index is None else index == 0


def selective(rows: list, threshold: float = 0.9) -> dict:
    policy = Policy(threshold, threshold)
    settled = [value == row["label"] for row in rows for status, value in (decision(row, policy),) if status == "ok"]
    return {"threshold": threshold, "accepted": len(settled), "correct": sum(settled), "total": len(rows),
            "coverage": len(settled) / len(rows) if rows else 0, "precision": sum(settled) / len(settled) if settled else None}


def summarize(res: dict, t: float = 0.9) -> dict:
    rows = [r for r in res["checks"] if r["p"] is not None and not r.get('error')]
    p = np.array([r["p"] for r in rows])
    y = np.array([r["label"] for r in rows], dtype=bool)
    usable = np.array([r["mass"] is None or r["mass"] >= 0.5 for r in rows], dtype=bool)
    settled = selective(res["checks"], t)
    right = (p >= 0.5) == y
    by = lambda key: {k: round(float(right[[r[key] == k for r in rows]].mean()), 2)
                      for k in sorted({r[key] for r in rows})}
    pairs: dict = {}
    for r, pr in zip(rows, p):
        pairs.setdefault(r["pair"], []).append(pr if r["label"] else -pr)
    pk = [r for r in res["picks"] if r["scores"] and not r.get('error')]
    pick_ok = [max(r["scores"], key=r["scores"].get) == r["answer"] for r in pk]
    submitted = len(res["checks"])
    complete_pairs = [values for values in pairs.values() if len(values) == 2]
    return {"suite": SUITE_VERSION, "fixture_id": res.get("fixture_id"), "role": res.get("role"), "predictor": res.get("predictor"), "population": res.get("population", "identified development sample"), "timing": res.get("timing", "public per-request latency"), "items": submitted, "valid_scores": len(rows),
            "accuracy_given_prediction": round(float(right.mean()), 3) if rows else None,
            "correct_per_submission": round(int(right.sum()) / submitted, 3) if submitted else None,
            "pair_ordering": round(float(np.mean([sum(values) > 0 for values in complete_pairs])), 3) if complete_pairs else None,
            "valid_pairs": len(complete_pairs),
            "auc": round(_auc(p, y), 3) if y.any() and (~y).any() else None,
            "brier": round(float(((p - y) ** 2).mean()), 3) if rows else None,
            "ece": round(_ece(p, y), 3) if rows else None,
            f"coverage_{t}": round(settled["coverage"], 3) if submitted else None,
            f"precision_{t}": None if settled["precision"] is None else round(settled["precision"], 3),
            "pick_submitted": len(res['picks']), "pick_valid_scores": len(pk),
            "pick_valid_score_accuracy": round(float(np.mean(pick_ok)), 3) if pk else None,
            "pick_accuracy": round(sum(pick_ok) / len(res['picks']), 3) if res['picks'] else None,
            "ms_p50": round(float(np.median([r["ms"] for r in res["checks"]])), 1) if submitted else None,
            "check_statuses": dict(Counter(row.get('status', 'unexamined') for row in res['checks'])),
            "pick_statuses": dict(Counter(row.get('status', 'unexamined') for row in res['picks'])),
            "errors": sum(r["p"] is None and r["error"] is not None for r in res["checks"]),
            "low_mass": int((~usable).sum()), "by_tag": by("tag"), "by_lang": by("lang")}


DISTRACTORS = [
    "cooking recipes and ingredients", "the weather forecast", "football and other sports",
    "music, concerts and albums", "gardening and plants", "pets and animal care", "travel and hotel bookings",
    "movies and television series", "home renovation and furniture", "fitness and gym workouts",
    "astronomy and space", "car maintenance", "school homework", "real estate and renting", "fashion and clothing",
]
POSITION_SLOTS = (0, 9, 19)


SCENARIOS = [
    ("Our team has two engineers and little time for operations.",
     {"pg": "a single managed PostgreSQL instance", "cass": "a self-hosted five-node Cassandra cluster",
      "sheet": "a shared spreadsheet"},
     ("Which option best fits a team that prioritises simple maintenance?", "pg"),
     ("Which option best handles millions of writes per second across regions?", "cass")),
    ("A customer writes: my order arrived broken and I am furious.",
     {"refund": "Reply: we have issued a full refund to your card.",
      "sorry": "Reply: I'm very sorry; a replacement ships today at no cost.",
      "policy": "Reply: please read our returns policy page."},
     ("Which reply best calms the customer without promising money back?", "sorry"),
     ("Which reply best resolves it if the customer only wants their money back?", "refund")),
    ("We need to commute 3 km in a city with bike lanes; it is sunny.",
     {"bike": "cycling", "car": "driving a car", "plane": "taking a plane"},
     ("Which option is the cheapest and healthiest?", "bike"),
     ("Which option keeps us dry if a storm starts?", "car")),
    ("A 10-person startup wants to launch a web app next week.",
     {"rails": "a monolith with a mainstream web framework", "micro": "forty microservices on Kubernetes",
      "asm": "hand-written assembly"},
     ("Which option ships fastest with the least operational work?", "rails"),
     ("Which option lets forty independent teams deploy separately?", "micro")),
    ("Um cliente pergunta como cancelar a assinatura.",
     {"link": "Resposta: o link para cancelar está em Configurações > Plano.",
      "offer": "Resposta: antes de cancelar, oferecemos 50% de desconto por 3 meses.",
      "ignore": "Resposta: obrigado pelo contato."},
     ("Qual resposta atende o pedido do cliente de forma mais direta?", "link"),
     ("Qual resposta tenta reter o cliente?", "offer")),
    ("We must store 5 TB of photos that are rarely read.",
     {"cold": "object storage in a cold, cheap tier", "ram": "an in-memory cache", "ssd": "fast local SSDs"},
     ("Which option minimises monthly cost?", "cold"),
     ("Which option gives the lowest read latency regardless of cost?", "ram")),
]


def positions(rt) -> dict:
    by = {s + 1: 0 for s in POSITION_SLOTS}
    flips = 0
    for text, answer in PICKS:
        others = [d for k, d in TOPICS.items() if k != answer] + DISTRACTORS
        tops = []
        for slot in POSITION_SLOTS:
            options = others[:slot] + [TOPICS[answer]] + others[slot:]
            r = rt.pick(render(text), TOPIC_QUESTION, options, min_score=0.9)
            tops.append(max(range(len(options)), key=lambda i: r.scores[i]) == slot)
            by[slot + 1] += tops[-1]
        flips += len(set(tops)) > 1
    return {"top1_by_slot": {s: round(n / len(PICKS), 3) for s, n in by.items()}, "order_flips": flips,
            "items": len(PICKS)}


def comparative(rt) -> dict:
    follows, stable, n = 0, 0, 0
    for state, options, *goals in SCENARIOS:
        tops = []
        for question, want in goals:
            pair = [max(r.scores, key=r.scores.get) for r in
                    (rt.pick(state, question, {k: options[k] for k in order})
                     for order in (list(options), list(reversed(options))))]
            stable += pair[0] == pair[1]
            follows += sum(t == want for t in pair)
            n += 2
            tops.append(pair[0])
    return {"goal_followed": round(follows / n, 3), "order_stable": round(stable / (n // 2), 3), "items": n}


def probe(rt, role: str = "judge") -> dict:
    summary = summarize(run(rt, role)) | {"positions": positions(_role_runtime(rt, role)), "comparative": comparative(_role_runtime(rt, role))}
    files.write_cached("quality", rt.cache_key(role), summary | {"at": files.now()})
    return summary
