"""Biblioteca de entidades do lote de schemas v3 (usada por build_schemas_v3.py).

DSL = a de build_schemas.py (name=type;...) + extensoes v3:
  ^            sufixo: campo RARO (presente em 3-8% dos documentos)
  refu:coll    referencia com heterogeneidade (ObjectId/UUID/string do _id alvo em ~12% dos docs como string)
  tu           data (Date) com ~10% dos documentos como string ISO-8601 (dado migrado)
  ds           data SEMPRE como string ISO-8601 (legado)
  nu:lo..hi    inteiro que em ~8% dos docs vem como string numerica
  geo          GeoJSON Point
  mp.pool(T)   mapa dinamico (chaves variaveis de `pool`, valores do tipo T)
  aa:T         array de arrays de T
Indices: !a unico | ttl:a | text:a | ~a@flag parcial | sp:a sparse | ci:a collation en/strength 2 | hash:a hashed |
         wild:a wildcard (a.$**) | geo:a 2dsphere | "a,-b" composto.
Entidade: EE(chave, chave_pt, campos, indices, hint="") ; hint: "ts:<campo_tempo>:<campo_meta>" (time-series) | "cap" (capped).
"""


def EE(key, pt_key, fields, idx="", hint=""):
    return (key, pt_key, fields, idx, hint)


# ============================================================================ NOVOS DOMINIOS
TELECOM = [
 EE("subscribers", "assinantes", "msisdn=code:55119;imsi=hex;full_name=name;plan_id=refu:plans;status=e:active|suspended|churned|porting;activated_at=tu;birth_date=t:1940-01-01..2007-12-31^;contact=o(email=email?,phone=phone,address=o(city=city,state=e:SP|RJ|MG|BA|RS|PR,geo=geo)?);billing=o(cycle_day=i:1..28,method=e:card|boleto|debit,autopay=b:0.6,tax=o(tax_id=code:TX,regime=e:simples|presumido,fiscal=o(ie=code:IE?,im=code:IM?)?)?)?;sim=o(iccid=hex,type=e:physical|esim,vendor=o(name=word,batch=o(lot=code:LT,mfg_date=t))?)?;tags=as:word?;consents=mp.word(b:0.5)?",
    "!msisdn;plan_id,status;sp:birth_date;ci:full_name;hash:imsi;wild:consents"),
 EE("plans", "planos", "code=slug;name=title;tier=e:prepaid|postpaid|control|enterprise;price_monthly=m:5..400;data_gb=nu:1..500;features=mp.word(o(enabled=b:0.8,limit=i:1..100?,notes=sentence^));throttling=o(after_gb=i:1..300,kbps=i:64..2048)?;valid_from=t;valid_to=t??;regions=as:e:N|NE|CO|SE|S;promo=o(code=code:PR,discount_pct=i:5..50,stack=as:code:PR)^",
    "!code;tier,-price_monthly;regions"),
 EE("cdr", "registros_chamada", "subscriber_id=refu:subscribers;a_number=code:5511;b_number=code:5521;start=t;duration_s=i:0..7200;kind=e:voice|sms|data|mms|roaming;route=o(carrier=e:vivo|claro|tim|oi,trunk=code:TK);cell=o(cell_id=code:CL,lac=i:1..9000,tower=o(site=code:ST,sector=i:1..6,loc=geo));charge=m:0..90;bytes_up=l:0..900000000;bytes_down=l:0..900000000;rating=o(zone=e:local|national|intl,tariff=code:TF,discounts=a(code=code:DC,amount=m:0..20)?)?",
    "route.carrier,-start;subscriber_id,-start", "ts:start:route"),
 EE("towers", "torres", "code=code:ST;name=title;location=geo;tech=as:e:2G|3G|4G|5G;sectors=a(sector_id=i:1..6,azimuth=i:0..359,bands=as:i:700..3700,status=e:up|down|maint);owner=word;installed_at=t;power=o(grid=b:0.9,battery=o(kwh=d:1..40,health=d:0..1)?,generator=o(fuel_l=i:0..2000)?)?;capacity_by_band=mp.word(i:10..900)",
    "!code;geo:location;tech;wild:capacity_by_band"),
 EE("invoices", "faturas", "subscriber_id=refu:subscribers;number=code:INV;period=o(from=t,to=t);lines=a(kind=e:plan|usage|addon|tax|discount,desc=sentence,amount=m:-50..400,meta=mp.word(word)?);total=m:0..2000;due=tu;paid_at=t??;status=e:open|paid|overdue|disputed;dunning=a(step=i:1..5,at=t,channel=e:sms|email|call)?",
    "!number;subscriber_id,-due;status,paid_at;~due@status"),
 EE("tickets", "chamados", "subscriber_id=refu:subscribers;channel=e:app|call|store|chat;topic=e:billing|coverage|device|porting|fraud;opened_at=t;closed_at=t??;priority=e:p1|p2|p3;thread=a(at=t,from=e:agent|customer|bot,text=sentence,attachments=as:o(name=slug,bytes=l:100..9000000)?);sla=o(due_at=t,breached=b:0.1,history=a(at=t,state=word))?;csat=i:1..5^",
    "subscriber_id,-opened_at;topic,priority;thread.from"),
 EE("topups", "recargas", "subscriber_id=refu:subscribers;amount=m:5..200;channel=e:app|pix|store|atm;at=t;promo=o(code=code:PR,bonus_pct=i:0..100)^;receipt=o(nsu=hex,auth=code:AU)", "subscriber_id,-at;channel"),
 EE("devices", "aparelhos", "imei=code:35;brand=e:samsung|apple|motorola|xiaomi|lg;model=word;status=e:active|lost|blocked|refurb;subscriber_id=refu:subscribers?;locks=o(carrier=b:0.2,theft=b:0.03,history=a(at=t,reason=word))^;specs=mp.word(word);first_seen=tu", "!imei;subscriber_id;brand,status;sp:locks.theft"),
 EE("network_events", "eventos_rede", "at=t;severity=e:info|minor|major|critical;element=code:NE;site=ref:towers?;alarm=o(code=code:AL,text=sentence,details=o(counters=mp.metric(l:0..1000000),thresholds=mp.metric(d:0..100)?))?;cleared=b:0.7", "-at;severity,element;ttl:at", "cap"),
 EE("roaming_agreements", "acordos_roaming", "partner=title;country=country;rates=mp.cur(o(voice=m:0..9,sms=m:0..2,data_mb=m:0..2));windows=aa:t^;active=b:0.8;signed=ds;tiers=a(name=word,min_mb=i:0..1000,max_mb=i:1000..99999)", "!partner;country,active"),
 EE("portability_requests", "portabilidades", "msisdn=code:55119;donor=e:vivo|claro|tim|oi;recipient=e:vivo|claro|tim|oi;state=e:requested|scheduled|done|rejected;history=a(at=t,state=word,by=word?);requested_at=tu;window=o(from=t,to=t)^", "msisdn;state,-requested_at"),
 EE("campaigns", "campanhas", "name=title;segment=o(rule=o(all=a(field=word,op=e:eq|gt|in|lt,value=word),any=a(field=word,op=e:eq|gt|in|lt,value=word)?));budget=m:1000..900000;channels=as:e:sms|push|email|call;stats=mp.day(o(sent=i:0..9000,opened=i:0..4000,converted=i:0..500));starts=t;ends=t??;owner=name", "starts;channels;wild:stats"),
 EE("fraud_alerts", "alertas_fraude", "subscriber_id=refu:subscribers;score=d:0..1;signals=mp.metric(d:0..1);model=o(name=word,version=code:v,features=aa:d:0..1);raised_at=t;status=e:open|confirmed|dismissed;reviewer=name?", "subscriber_id,-raised_at;status,score"),
 EE("data_usage", "uso_dados", "subscriber_id=refu:subscribers;day=t;usage_mb=mp.hour(i:0..500);apps=mp.word(l:0..99999999);roaming=b:0.07;throttled=b:0.1", "subscriber_id,-day;roaming"),
 EE("dealers", "revendas", "code=code:DL;name=title;channel=e:own|franchise|online;location=geo;staff=a(name=name,role=e:seller|manager,sales_by_month=aa:i:0..99)?;open=b:0.9", "!code;geo:location;channel"),
]

ADTECH = [
 EE("advertisers", "anunciantes", "name=title;legal_id=code:TX;vertical=e:retail|finance|auto|telecom|gaming|travel;billing=o(currency=e:BRL|USD|EUR,terms_days=i:0..90,tax=o(rate=d:0..0.3,exempt=b:0.05)?)?;created_at=tu;status=e:active|paused|closed;contacts=a(name=name,email=email,role=e:billing|tech|exec)", "!legal_id;vertical,status;ci:name"),
 EE("campaigns", "campanhas", "advertiser_id=refu:advertisers;name=title;objective=e:awareness|traffic|conversion|app_install;status=e:draft|live|paused|ended;budget=o(total=m:500..900000,daily=m:10..30000,currency=e:BRL|USD,pacing=e:even|asap|smart);flight=o(start=t,end=t);targeting=o(geo=as:country,devices=as:e:mobile|desktop|tv|ctv,age=o(min=i:13..40,max=i:41..90)?,languages=mp.lang(b:0.5)?);bid_strategy=o(kind=e:cpc|cpm|cpa|roas,target=d:0.1..90,caps=o(max_bid=d:0.5..20,min_bid=d:0..1)?)", "advertiser_id,status;flight.start,flight.end;objective,-budget.total"),
 EE("ad_groups", "grupos_anuncio", "campaign_id=refu:campaigns;name=slug;status=e:live|paused|ended;bid_modifiers=mp.slot(d:0.1..3);keywords=as:word;placements=as:ref:placements?;created_at=t", "campaign_id,status;keywords"),
 EE("creatives", "criativos", "ad_group_id=ref:ad_groups;format=e:banner|video|native|html5|audio;size=o(w=i:50..1920,h=i:50..1080);assets=a(kind=e:image|video|text,url=url,bytes=l:1000..90000000,meta=mp.word(word)?);status=e:approved|pending|rejected;review=o(by=name?,at=t,notes=sentence?)?;landing_url=url;version=i:1..9", "ad_group_id,status;format,-version"),
 EE("impressions", "impressoes", "at=t;campaign_id=ref:campaigns;creative_id=ref:creatives;publisher_id=ref:publishers;device=o(type=e:mobile|desktop|tv|ctv,os=e:android|ios|windows|macos|linux,ua_hash=hex,geo=o(country=country,city=city,loc=geo)?);cost_micros=l:100..9000000;viewable=b:0.62;bid=o(price=d:0.05..30,win_price=d:0.05..30,auction=e:first|second);user=o(uid=u4,segments=as:code:SG)", "campaign_id,-at;publisher_id,at;device.os", "ts:at:device"),
 EE("clicks", "cliques", "impression_id=ref:impressions?;campaign_id=ref:campaigns;at=t;click_id=u4;position=o(x=i:0..1920,y=i:0..1080);fraud=o(score=d:0..1,reasons=as:word?)?;landing=o(url=url,referrer=url?,utm=mp.word(word)?)", "!click_id;campaign_id,-at;sp:fraud.score"),
 EE("conversions", "conversoes", "click_id=u4;campaign_id=ref:campaigns;kind=e:purchase|signup|install|lead;value=m:0..3000;at=t;attribution=o(model=e:last_click|linear|data_driven,window_days=i:1..90,touchpoints=a(campaign=ref:campaigns,at=t,weight=d:0..1))", "campaign_id,kind,-at;attribution.model"),
 EE("bid_events", "eventos_lance", "at=t;exchange=e:google|xandr|magnite|pubmatic|openx;request_id=u4;floor=d:0..5;bids=a(bidder=word,price=d:0..20,latency_ms=i:5..400,won=b:0.2,reasons=as:word?);result=e:won|lost|timeout|nobid;site=o(domain=slug,page=url,categories=as:code:IAB)", "-at;exchange,result;ttl:at", "cap"),
 EE("audiences", "audiencias", "name=title;kind=e:first_party|lookalike|retargeting|contextual;rules=o(include=a(field=word,op=e:eq|in|gt,value=word),exclude=a(field=word,op=e:eq|in|gt,value=word)?,lookback_days=i:1..180);size=l:100..90000000;refresh=o(every_h=i:1..168,last_at=t)?;owner=ref:advertisers;stats=mp.day(l:0..99999)", "owner,kind;-size;wild:stats"),
 EE("publishers", "editores", "domain=slug;name=title;category=e:news|sports|games|shopping|video|social;traffic_tier=e:t1|t2|t3;ads_txt=o(verified=b:0.8,lines=i:0..400,sellers=as:o(id=code:SL,rel=e:DIRECT|RESELLER))?;revenue_share=d:0.3..0.9;floor_prices=mp.cur(d:0.01..3);country=country", "!domain;category,traffic_tier;wild:floor_prices"),
 EE("placements", "posicoes", "publisher_id=ref:publishers;slot=code:SLOT;format=as:e:banner|video|native;sizes=aa:i:50..970;viewability=d:0.2..0.95;active=b:0.9;floor=d:0..4", "publisher_id,active;!slot"),
 EE("pacing", "ritmo", "campaign_id=ref:campaigns;day=t;spent=m:0..30000;delivered=l:0..9000000;target=l:1000..9000000;hourly=mp.hour(m:0..2000);underdelivery=b:0.2", "campaign_id,-day;underdelivery"),
 EE("daily_reports", "relatorios_diarios", "campaign_id=ref:campaigns;day=t;metrics=mp.metric(d:0..99999);by_device=mp.word(o(imps=l:0..900000,clicks=l:0..9000,cost=m:0..9000));roas=d:0..12^;currency=e:BRL|USD", "campaign_id,-day;currency"),
 EE("experiments", "experimentos", "name=title;campaign_id=ref:campaigns;variants=a(key=code:V,creative=ref:creatives,weight=d:0..1);metric=e:ctr|cvr|cpa|roas;status=e:running|done|aborted;result=o(winner=code:V?,lift=d:-0.3..0.5,p_value=d:0..1,samples=mp.word(l:100..900000))?;started_at=t", "campaign_id,status;metric"),
 EE("fraud_flags", "marcacoes_fraude", "publisher_id=ref:publishers;kind=e:ivt|bot|click_farm|domain_spoof;severity=i:1..5;evidence=o(sample=as:u4,rates=mp.metric(d:0..1),notes=sentence?);flagged_at=t;resolved=b:0.4", "publisher_id,kind;severity,-flagged_at"),
 EE("attribution_paths", "caminhos_atribuicao", "user=u4;paths=aa:code:CH;converted=b:0.15;value=m:0..2000^;first_touch=tu;last_touch=t;model_scores=mp.word(d:0..1)", "user;converted,-last_touch"),
]

RETAIL_POS = [
 EE("stores", "lojas", "code=code:LJ;name=title;format=e:hyper|super|express|outlet;address=o(street=title,city=city,state=e:SP|RJ|MG|PR|SC|RS,geo=geo);opened_at=t;hours=mp.word(o(open=code:H,close=code:H)?);manager=name;region=e:N|NE|CO|SE|S;features=as:word", "!code;region,format;geo:address.geo"),
 EE("terminals", "terminais", "store_id=refu:stores;serial=hex;kind=e:pos|selfcheckout|mobile;firmware=code:FW;last_seen=tu;status=e:online|offline|maint;peripherals=a(kind=e:printer|scanner|scale|pinpad,vendor=word,model=code:MD)", "store_id,status;!serial"),
 EE("cashiers", "operadores", "badge=code:OP;name=name;store_id=refu:stores;role=e:cashier|supervisor|trainee;hired_at=t;permissions=mp.word(b:0.5);active=b:0.9;pin_hash=hex", "!badge;store_id,role"),
 EE("sales", "vendas", "receipt_no=code:NF;store_id=refu:stores;terminal_id=ref:terminals;cashier_id=ref:cashiers;at=tu;lines=a(sku=sku,qty=nu:1..12,unit_price=m:0.5..900,discount=o(kind=e:pct|abs|promo,value=d:0..50,promo_id=ref:promotions?)?,tax=o(code=code:TX,rate=d:0..0.3));payments=a(method=e:cash|credit|debit|pix|voucher,amount=m:1..3000,card=o(brand=e:visa|master|elo,last4=code:,nsu=hex,installments=i:1..12?)?);total=m:1..5000;customer=o(cpf_hash=hex,loyalty=o(card=code:LC,points=i:0..9000)?)?;status=e:closed|voided|suspended", "!receipt_no;store_id,-at;lines.sku;payments.method,total"),
 EE("shifts", "turnos", "cashier_id=ref:cashiers;terminal_id=ref:terminals;opened_at=t;closed_at=t??;float_open=m:0..500;counted=mp.word(m:0..9000);variance=m:-80..80;supervisor_ok=b:0.9", "cashier_id,-opened_at;terminal_id"),
 EE("cash_events", "eventos_caixa", "at=t;terminal_id=ref:terminals;kind=e:open|sangria|suprimento|void|nosale|close;amount=m:0..3000;approved_by=ref:cashiers?;note=sentence^", "-at;kind,terminal_id;ttl:at", "cap"),
 EE("items", "itens", "sku=sku;ean=code:789;description=title;dept=e:grocery|bakery|dairy|beauty|home|electronics;price=m:0.5..900;cost=m:0.3..700;unit=e:un|kg|l|pack;taxes=o(icms=d:0..0.25,pis=d:0..0.05,cofins=d:0..0.1);attrs=mp.word(word);suppliers=as:ref:suppliers;active=b:0.92", "!sku;!ean;dept,price;wild:attrs"),
 EE("price_changes", "alteracoes_preco", "sku=sku;store_id=refu:stores?;old_price=m:0.5..900;new_price=m:0.5..900;effective=t;reason=e:promo|cost|competitor|clearance;approved=o(by=name,at=t)^", "sku,-effective;reason"),
 EE("promotions", "promocoes", "code=code:PM;name=title;kind=e:bogo|pct|bundle|coupon;rules=o(min_qty=i:1..10,skus=as:sku,stores=as:ref:stores?,customers=e:all|loyalty|new);starts=t;ends=t;stackable=b:0.2;redeemed=mp.day(i:0..900)", "!code;starts,ends;kind"),
 EE("stock_counts", "contagens", "store_id=refu:stores;sku=sku;counted_at=t;system_qty=i:0..400;counted_qty=i:0..400;operator=ref:cashiers?;reason=e:cycle|audit|damage|theft^", "store_id,sku,-counted_at"),
 EE("returns", "devolucoes", "sale_id=ref:sales;at=t;lines=a(sku=sku,qty=i:1..4,reason=e:defect|regret|wrong);refund=m:1..2000;method=e:original|voucher|cash;approved_by=ref:cashiers?", "sale_id;-at"),
 EE("z_reports", "reducao_z", "store_id=refu:stores;terminal_id=ref:terminals;date=t;totals=mp.word(m:0..90000);tax_summary=mp.word(o(base=m:0..90000,tax=m:0..9000));counters=o(sales=i:10..900,voids=i:0..30,returns=i:0..30)", "store_id,-date;terminal_id,date"),
 EE("suppliers", "fornecedores", "name=title;tax_id=code:TX;lead_days=i:1..60;terms=e:net30|net60|cash;contact=o(name=name,email=email,phone=phone?);active=b:0.9", "!tax_id;active"),
 EE("purchase_orders", "pedidos_compra", "supplier_id=ref:suppliers;store_id=refu:stores;issued_at=tu;lines=a(sku=sku,qty=i:1..900,cost=m:0.3..700,received=i:0..900?);status=e:open|partial|received|cancelled;eta=t??", "supplier_id,-issued_at;status;lines.sku"),
]

MANUFACTURING = [
 EE("plants", "fabricas", "code=code:PL;name=title;location=geo;country=country;lines_count=i:1..40;certifications=as:e:ISO9001|ISO14001|IATF16949|ISO45001;shifts=mp.word(o(start=code:H,end=code:H,crew=i:5..200));commissioned=t;energy=o(grid_kw=i:500..90000,solar=o(kwp=i:0..5000)?)?", "!code;country;geo:location"),
 EE("production_lines", "linhas_producao", "plant_id=refu:plants;code=code:LN;product_family=e:engines|gearboxes|pcb|plastics|cables;takt_s=d:5..600;stations=a(seq=i:1..40,name=word,machine_id=ref:machines?,cycle_s=d:1..300);capacity_per_h=i:10..9000;status=e:running|idle|maint|changeover", "plant_id,status;!code;stations.machine_id"),
 EE("machines", "maquinas", "asset_tag=code:MC;line_id=ref:production_lines;kind=e:cnc|press|robot|oven|injector|agv;vendor=word;model=code:MD;commissioned=tu;spec=o(power_kw=d:1..900,axes=i:1..9^,tolerances=mp.word(d:0.001..1),controller=o(vendor=word,fw=code:FW,params=mp.word(word)?));status=e:running|idle|down|maint;oee=o(availability=d:0.3..1,performance=d:0.3..1,quality=d:0.5..1)", "!asset_tag;line_id,kind;status,oee.availability;wild:spec.tolerances"),
 EE("work_orders", "ordens_producao", "wo_no=code:WO;product_id=ref:parts;line_id=ref:production_lines;qty_planned=i:10..9000;qty_done=i:0..9000;status=e:planned|released|running|done|cancelled;planned=o(start=t,end=t);actual=o(start=t,end=t??)?;routing=a(op=i:10..90,station=word,std_min=d:0.5..120,tools=as:ref:tools?);lot=ref:batches?;priority=i:1..5;customer_ref=nu:100000..999999^", "!wo_no;line_id,status;planned.start;priority,status"),
 EE("boms", "listas_materiais", "parent=ref:parts;revision=code:R;components=a(part=ref:parts,qty=d:0.1..40,uom=e:un|kg|m|l,alt=as:ref:parts?,children=a(part=ref:parts,qty=d:0.1..10,scrap_pct=d:0..0.1)?);effective=t;obsolete=b:0.1;approved=o(by=name,at=t)", "parent,-revision;components.part;obsolete"),
 EE("parts", "pecas", "part_no=code:PN;description=title;kind=e:raw|component|assembly|finished;uom=e:un|kg|m|l;cost=m:0.1..9000;supplier_id=ref:suppliers?;params=mp.word(word);hazmat=o(class=word,un_no=code:UN)^;lead_days=i:0..120;drawings=a(rev=code:R,url=url,format=e:pdf|dwg|step)?", "!part_no;kind,-cost;supplier_id;wild:params"),
 EE("inspections", "inspecoes", "wo_id=ref:work_orders;part_id=ref:parts;at=tu;inspector=name;kind=e:incoming|inprocess|final|audit;measurements=a(dim=word,nominal=d:0..500,actual=d:0..500,tol=d:0.001..2,pass=b:0.93);result=e:pass|fail|rework;sample=o(size=i:1..200,aql=d:0.065..4)", "wo_id,-at;result,kind;measurements.pass"),
 EE("defects", "defeitos", "inspection_id=ref:inspections;code=code:DF;severity=e:minor|major|critical;cause=o(category=e:man|machine|material|method,detail=sentence?,five_whys=as:sentence?)?;qty=i:1..90;cost=m:0..9000;images=as:url^;at=t", "inspection_id;severity,-at;cause.category"),
 EE("downtime", "paradas", "machine_id=ref:machines;at=t;duration_min=d:0.5..600;reason=e:breakdown|changeover|material|quality|planned|no_operator;line=o(code=code:LN,plant=code:PL);details=o(error_code=code:ER?,stop_codes=as:code:SC)", "machine_id,-at;reason", "ts:at:line"),
 EE("machine_readings", "leituras_maquina", "at=t;machine_id=ref:machines;src=o(machine=code:MC,channel=e:vib|temp|current|pressure);value=d:-50..900;unit=e:C|A|bar|mm_s|rpm;quality=i:0..100;alarms=as:code:AL^", "machine_id,-at;src.channel", "ts:at:src"),
 EE("maintenance_orders", "ordens_manutencao", "machine_id=ref:machines;kind=e:preventive|corrective|predictive;opened=t;closed=t??;tasks=a(step=i:1..20,text=sentence,done=b:0.7,parts=a(part=ref:parts,qty=i:1..9)?);cost=m:10..40000;technician=name;downtime_min=d:0..900^", "machine_id,-opened;kind,closed"),
 EE("operators", "operadores", "badge=code:OP;name=name;plant_id=refu:plants;skills=mp.word(i:1..5);certs=a(name=word,valid_until=t);shift=e:A|B|C;active=b:0.92", "!badge;plant_id,shift;wild:skills"),
 EE("batches", "lotes", "lot_no=code:LT;part_id=ref:parts;qty=i:1..90000;produced=t;expires=t??;qc=o(status=e:released|hold|rejected,tests=mp.word(o(value=d:0..100,spec=word)),released_by=name?);genealogy=aa:code:LT^", "!lot_no;part_id,produced;qc.status"),
 EE("suppliers", "fornecedores", "name=title;tax_id=code:TX;rating=d:1..5;country=country;certs=as:word;scorecard=o(delivery=d:0..1,quality=d:0..1,cost=d:0..1);lead_days=i:1..90;contacts=a(name=name,email=email)?", "!tax_id;country,rating"),
 EE("tools", "ferramentas", "tool_no=code:TL;kind=e:drill|mill|insert|gauge|fixture;life=o(max_cycles=i:100..90000,used=i:0..90000);calibration=o(last=t,due=t,history=a(at=t,by=name,result=e:ok|adjusted|failed))?;location=word", "!tool_no;kind;calibration.due"),
 EE("shipments", "expedicoes", "wo_id=ref:work_orders;customer=title;shipped_at=tu;pallets=a(sscc=code:SS,items=a(part=ref:parts,qty=i:1..900));carrier=word;docs=mp.word(url);status=e:packed|shipped|delivered", "wo_id;customer,-shipped_at"),
]

# ============================================================================ EXTRAS v3 PARA DOMINIOS EXISTENTES
# (entidades adicionais, so v3; as chaves nao colidem com as da biblioteca ENT de build_schemas.py)
EXTRA = {
 "commerce": [
  EE("price_history", "historico_precos", "product_id=refu:products;at=t;price=m:5..900;currency=e:BRL|USD|EUR;changed_by=name^;source=e:erp|manual|import", "product_id,-at", "ts:at:product_id"),
  EE("search_logs", "logs_busca", "at=t;session=u4;query=sentence;filters=mp.word(word);results=i:0..400;clicked=as:ref:products?;device=o(type=e:mobile|desktop,os=word,viewport=aa:i:300..1920)", "-at;session;ttl:at", "cap"),
  EE("promotions", "promocoes", "code=code:PM;rules=o(all=a(sku=sku,min_qty=i:1..6),exclude=o(categories=as:word,brands=as:word)?);window=o(from=tu,to=tu);channels=mp.word(b:0.5);budget=m:100..90000", "!code;window.from"),
 ],
 "saas": [
  EE("usage_events", "eventos_uso", "at=t;tenant_id=refu:tenants;user_id=ref:users?;event=e:page|click|api|export|login|invite;props=mp.word(word);ctx=o(client=o(app=word,version=code:v,os=o(name=word,ver=code:v)),geo=o(country=country,city=city)?,session=u4);duration_ms=nu:1..90000", "tenant_id,-at;event", "ts:at:ctx"),
  EE("integrations", "integracoes", "tenant_id=refu:tenants;provider=e:slack|github|jira|salesforce|stripe|zapier;config=mp.word(word);oauth=o(scopes=as:word,expires_at=tu,refresh=o(rotated_at=t,count=i:0..90)?)?;status=e:ok|degraded|revoked;last_sync=t??", "tenant_id,provider;status"),
  EE("roles", "papeis", "tenant_id=ref:tenants;name=slug;permissions=mp.word(as:e:read|write|admin|share);inherits=as:ref:roles?;builtin=b:0.3;created_by=ref:users", "tenant_id,name;wild:permissions"),
  EE("billing_events", "eventos_cobranca", "tenant_id=refu:tenants;at=tu;type=e:charge|refund|dispute|credit|proration;amount=m:-500..9000;provider=o(name=word,id=hex,raw=o(status=word,fees=mp.cur(m:0..30)))", "tenant_id,-at;type"),
 ],
 "logistics": [
  EE("gps_pings", "pings_gps", "vehicle_id=refu:vehicles;at=t;loc=geo;speed_kmh=d:0..130;heading=i:0..359;ign=b:0.8;meta=o(src=e:obd|phone|tracker,hdop=d:0..9?)", "vehicle_id,-at;geo:loc", "ts:at:vehicle_id"),
  EE("customs", "alfandega", "shipment_id=ref:shipments;declared=mp.word(o(hs=code:HS,value=m:1..90000,qty=i:1..900));status=e:filed|held|cleared|rejected;events=a(at=tu,code=code:CS,note=sentence?);broker=name?", "shipment_id;status"),
  EE("sla_contracts", "contratos_sla", "carrier_id=ref:carriers;lanes=a(from=city,to=city,hours=i:6..400,penalty=o(pct=d:0..0.3,cap=m:100..9000)?);valid=o(from=t,to=t);exceptions=mp.word(sentence)^", "carrier_id;valid.to"),
 ],
 "iot": [
  EE("metrics_1m", "metricas_1m", "device_id=ref:devices;at=t;vals=mp.metric(d:0..999);tags=o(site=code:ST,env=e:prod|test,fleet=word);n=i:1..60", "device_id,-at;tags.env", "ts:at:tags"),
  EE("ota_jobs", "jobs_ota", "firmware_id=ref:firmware;targets=a(device=ref:devices,state=e:queued|downloading|applied|failed|rolledback,progress=i:0..100,error=o(code=code:ER,msg=sentence)?);started=tu;policy=o(batch=i:1..500,window=o(from=code:H,to=code:H),retry=o(max=i:0..5,backoff_s=nu:1..900)?)", "firmware_id;started;targets.state"),
  EE("rules", "regras", "name=slug;scope=o(site=ref:sites?,model=e:th-100|pm-200|gw-300|cam-9?,labels=mp.word(word)?);condition=o(any=a(metric=e:temperature|humidity|co2,op=e:gt|lt,threshold=d:-20..120,for_s=i:0..3600),window=o(size_s=i:10..3600,agg=e:avg|max|min))?;actions=a(kind=e:alert|command|webhook,params=mp.word(word));enabled=b:0.8;version=i:1..30", "name;enabled"),
 ],
 "gaming": [
  EE("events", "eventos", "at=t;player_id=refu:players;type=e:login|kill|death|purchase|levelup|quit|chat;session=u4;payload=mp.word(word);pos=o(x=d:-500..500,y=d:-500..500,zone=word)?;build=code:v", "player_id,-at;type", "ts:at:build"),
  EE("matchmaking_queue", "fila_partidas", "player_id=ref:players;mode=e:ranked|casual|custom|coop;mmr=i:600..3200;enqueued_at=t;region=e:sa|na|eu|asia;party=as:ref:players?;prefs=mp.word(b:0.5)", "mode,region,mmr;-enqueued_at;ttl:enqueued_at", "cap"),
  EE("loot_tables", "tabelas_loot", "name=slug;tiers=mp.word(a(item=ref:items,weight=d:0.001..1,qty=o(min=i:1..5,max=i:5..20)));conditions=o(level=o(min=i:1..50,max=i:50..99),zones=as:word)?;version=i:1..20", "!name;wild:tiers"),
  EE("clans_wars", "guerras_clas", "guild_a=ref:guilds;guild_b=ref:guilds;rounds=a(n=i:1..9,winner=ref:guilds,score=aa:i:0..99,events=a(at=t,kind=word,by=ref:players?)?);started=t;ended=t??;prize=o(items=as:ref:items?,gold=l:0..9000000)", "guild_a,-started;ended"),
 ],
 "finance": [
  EE("market_ticks", "cotacoes_tick", "symbol=code:TK;at=t;bid=d:1..900;ask=d:1..900;last=d:1..900;vol=l:0..9000000;venue=e:B3|NYSE|NASDAQ|LSE;flags=as:word^", "symbol,-at;venue", "ts:at:symbol"),
  EE("kyc_cases", "casos_kyc", "customer_id=refu:customers;opened=tu;risk=o(score=i:0..100,factors=mp.word(d:0..1),band=e:low|medium|high);docs=a(kind=e:rg|cnh|passport|proof,status=e:ok|pending|rejected,ocr=mp.word(word)?);analyst=name?;decision=e:approved|rejected|pending??", "customer_id,-opened;risk.band,decision"),
  EE("statements", "extratos", "account_id=ref:accounts;period=o(year=i:2025..2026,month=i:1..12);opening=m:-500..90000;closing=m:-500..90000;entries=a(day=i:1..31,desc=sentence,amount=m:-9000..9000,tags=as:word?);fees=mp.word(m:0..90)", "account_id,period.year,period.month;!account_id"),
 ],
 "healthcare-synthetic": [
  EE("vitals_stream", "sinais_vitais", "patient_id=refu:patients;at=t;vals=mp.metric(d:0..200);device=o(serial=hex,model=word,fw=code:FW);alert=b:0.04", "patient_id,-at;alert", "ts:at:device"),
  EE("clinical_notes", "notas_clinicas", "visit_id=ref:visits;author=ref:practitioners;at=tu;sections=mp.word(sentence);codes=a(system=e:ICD10|SNOMED|LOINC,code=code:CD,display=title?);addenda=a(at=t,by=name,text=sentence)^;signed=b:0.9", "visit_id,-at;codes.code"),
  EE("referrals", "encaminhamentos", "patient_id=ref:patients;from=ref:practitioners;to=ref:practitioners;specialty=e:cardio|derm|ortho|neuro;status=e:sent|accepted|done|declined;history=a(at=t,state=word);urgency=i:1..4", "patient_id,-urgency;status"),
 ],
 "hr": [
  EE("badge_swipes", "registros_catraca", "employee_id=refu:employees;at=t;door=code:DR;direction=e:in|out;method=e:badge|face|pin;anomaly=b:0.02", "employee_id,-at;door", "ts:at:door"),
  EE("org_changes", "mudancas_org", "employee_id=ref:employees;effective=tu;changes=mp.word(o(from=word,to=word));approved_by=ref:employees?;reason=e:promotion|transfer|restructure|backfill", "employee_id,-effective;reason"),
  EE("surveys", "pesquisas", "title=title;audience=o(departments=as:ref:departments,levels=as:e:junior|mid|senior|lead);questions=a(id=code:Q,text=sentence,scale=aa:i:1..5?);responses=mp.day(i:0..400);anonymous=b:0.8", "title;audience.departments"),
 ],
 "content": [
  EE("page_views", "visualizacoes", "at=t;article_id=ref:articles;visitor=u4;referrer=mp.word(word)?;device=o(type=e:mobile|desktop|tablet,browser=o(name=word,ver=code:v));scroll_pct=nu:0..100;geo=o(country=country,region=word?)", "article_id,-at;device.type", "ts:at:device"),
  EE("translations", "traducoes", "article_id=ref:articles;lang=e:pt|en|es|fr|de;fields=mp.word(sentence);status=e:draft|review|live;translator=name?;memory=aa:code:TM^", "article_id,lang;status"),
  EE("ab_tests", "testes_ab", "article_id=refu:articles;variants=a(key=code:V,title=title,weight=d:0..1);metrics=mp.metric(d:0..1);winner=code:V??;started=tu", "article_id;started"),
 ],
 "travel": [
  EE("fare_quotes", "cotacoes_tarifa", "origin=code:AP;dest=code:AP;at=t;fares=mp.cur(m:50..9000);carrier=code:CA;cabin=e:Y|W|J|F;rules=o(refundable=b:0.3,change_fee=m:0..500,baggage=o(carry_on=i:0..2,checked=i:0..3))", "origin,dest,-at;carrier", "ts:at:carrier"),
  EE("payments", "pagamentos", "booking_id=ref:bookings;method=e:card|pix|voucher|points;amount=m:10..30000;status=e:authorized|captured|refunded|failed;gateway=o(name=word,tx=hex,response=mp.word(word));at=tu", "booking_id;status,-at"),
  EE("hotel_inventory", "estoque_hotel", "hotel_id=ref:hotels;date=t;rates=mp.word(m:50..2500);rooms_left=mp.word(i:0..40);restrictions=o(min_stay=i:1..7,closed=b:0.05)^", "hotel_id,date"),
 ],
 "workspace": [
  EE("activity_log", "log_atividades", "at=t;member_id=refu:members;kind=e:create|update|comment|move|delete|share;entity=o(type=e:task|doc|project,id=hex,path=aa:slug);diff=mp.word(o(from=word,to=word))?;client=o(app=e:web|ios|android,ver=code:v)", "member_id,-at;kind", "cap"),
  EE("automations", "automacoes", "project_id=refu:projects;name=title;trigger=o(type=e:status|due|assign,conditions=a(field=word,op=e:eq|in,value=word));actions=a(type=e:set|notify|create,params=mp.word(word));runs=mp.day(i:0..90);enabled=b:0.8", "project_id,enabled;wild:runs"),
  EE("time_entries", "apontamentos", "task_id=ref:tasks;member_id=ref:members;start=tu;minutes=nu:1..600;billable=b:0.6;tags=as:word?", "task_id,-start;member_id,billable"),
 ],
 "energy": [
  EE("meter_intervals", "intervalos_medidor", "meter_id=refu:meters;at=t;kwh=d:0..90;kvarh=d:0..30;voltage=d:100..260;quality=as:e:ok|est|sub^", "meter_id,-at", "ts:at:meter_id"),
  EE("dispatch_orders", "ordens_despacho", "unit_id=ref:generation_units;issued=tu;setpoints=aa:d:0..900;constraints=mp.word(d:0..999);status=e:sent|ack|done|rejected;operator=name", "unit_id,-issued;status"),
  EE("customer_complaints", "reclamacoes", "contract_id=ref:contracts;at=t;topic=e:billing|outage|meter|quality;thread=a(at=t,from=e:agent|customer,text=sentence);resolved=b:0.7;compensation=m:0..900^", "contract_id,-at;topic,resolved"),
 ],
 "education": [
  EE("attendance", "frequencia", "student_id=ref:students;course_id=ref:courses;date=tu;present=b:0.88;justification=o(kind=word,doc=url?)^;checkin=o(method=e:qr|manual|bio,at=t)?", "student_id,-date;course_id,date"),
  EE("grade_books", "diarios", "course_id=ref:courses;term=e:2025-1|2025-2|2026-1;weights=mp.word(d:0..1);entries=a(student=ref:students,scores=mp.word(d:0..10),final=d:0..10??)", "course_id,term"),
 ],
 "legal": [
  EE("docket_entries", "andamentos", "case_id=refu:cases;at=tu;code=code:AN;text=sentence;parties_notified=as:name;attachments=a(name=slug,pages=nu:1..200,hash=hex)^", "case_id,-at"),
  EE("time_records", "apontamentos_horas", "lawyer_id=ref:lawyers;case_id=ref:cases;day=t;minutes=nu:6..720;activity=mp.word(i:0..240);billed=b:0.7", "lawyer_id,-day;case_id,billed"),
 ],
}
